#include "etazero/inference.h"
#include "etazero/schema.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace etazero {
namespace {
std::vector<std::unique_ptr<Backend>> single(std::unique_ptr<Backend> backend) {
    std::vector<std::unique_ptr<Backend>> result; result.push_back(std::move(backend)); return result;
}
}
BatchEvaluator::BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                               size_t max_batch, size_t capacity, int wait_us)
    : BatchEvaluator(single(std::move(backend)), std::move(model), canvas, max_batch, capacity, wait_us, 0) {}
BatchEvaluator::BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends, std::string model, int canvas,
                               size_t max_batch, size_t capacity, int wait_us, size_t cache_entries)
    : backends_(std::move(backends)), cache_(cache_entries), model_(std::move(model)), max_batch_(max_batch),
      capacity_(capacity), input_size_(INPUT_PLANES * canvas * canvas + GLOBAL_FEATURES), wait_us_(wait_us) {
    if (backends_.empty() || model_.empty() || !max_batch || !capacity || wait_us < 0)
        throw std::runtime_error("Invalid inference service configuration");
    for (auto& b : backends_) if (!b) throw std::runtime_error("Null inference backend");
    rows_by_server.resize(backends_.size(),0);
    try {
        for (size_t i=0; i<backends_.size(); ++i) servers_.emplace_back(&BatchEvaluator::serve, this, i);
    } catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_); closing_=true; }
        changed_.notify_all(); for (auto& t:servers_) t.join(); throw;
    }
}
BatchEvaluator::~BatchEvaluator() { try { finish(); } catch (...) {} }
void BatchEvaluator::finish() {
    { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
    changed_.notify_all();
    for (auto& t:servers_) if (t.joinable()) t.join();
    if (failure_) std::rethrow_exception(failure_);
}
void BatchEvaluator::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    changed_.wait(lock,[&]{return failure_ || (queue_.empty() && active_==0);});
    if(failure_)std::rethrow_exception(failure_);
}
void BatchEvaluator::reset_stats() {
    drain();requests=0;batches=0;max_observed_batch=0;wait_microseconds=0;cache_hits=0;submitted=0;
    std::fill(rows_by_server.begin(),rows_by_server.end(),0);
}
Evaluation BatchEvaluator::evaluate(const std::vector<float>& obs) {
    if (obs.size() != input_size_) throw std::runtime_error("Inference input shape mismatch");
    // evaluate() is synchronous: the caller owns its input until completion.
    // One thread-local request replaces per-query shared_ptr/promise allocations.
    thread_local Request local;
    Request* request=&local;
    ++submitted;
    if (!cache_.empty()) {
        // Pack binary spatial planes and append exact global float bytes. The
        // evaluator is bound to one model, precision, and canvas.
        const size_t spatial = obs.size() - GLOBAL_FEATURES;
        request->cache_key.assign((spatial+7)/8, '\0');
        for (size_t i=0;i<spatial;++i) {
            if (obs[i]!=0 && obs[i]!=1) throw std::runtime_error("Cache requires binary input features");
            if (obs[i]) request->cache_key[i/8] |= static_cast<char>(1 << (7-i%8));
        }
        request->cache_key.append(reinterpret_cast<const char*>(obs.data()+spatial), GLOBAL_FEATURES*sizeof(float));
        uint64_t hash=14695981039346656037ULL;
        for (unsigned char c:request->cache_key) { hash^=c;hash*=1099511628211ULL; }
        // Avalanche before indexing: packed sparse planes otherwise leave poor
        // low-bit distribution in power-of-two tables (early moves use high bits).
        hash^=hash>>33;hash*=0xff51afd7ed558ccdULL;
        hash^=hash>>33;hash*=0xc4ceb9fe1a85ec53ULL;hash^=hash>>33;
        request->cache_slot=hash%cache_.size();
        std::shared_ptr<const Evaluation> found;
        {
            std::lock_guard<std::mutex> lock(cache_locks_[request->cache_slot%cache_locks_.size()]);
            auto& e=cache_[request->cache_slot];
            if (e.key==request->cache_key) found=e.output;
        }
        if (found) {
            { std::lock_guard<std::mutex> lock(mutex_);
              if (failure_) std::rethrow_exception(failure_);
              if (closing_) throw std::runtime_error("Inference service is closing"); }
            ++cache_hits;return *found;
        }
    }
    static std::atomic<uint64_t> next{0};
    request->id=next.fetch_add(1);request->owner=this;request->observation=&obs;
    request->submitted = std::chrono::steady_clock::now();
    {std::lock_guard<std::mutex> lock(request->mutex);request->done=false;request->error=nullptr;}
    {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return queue_.size() < capacity_ || closing_ || failure_; });
        if (failure_) std::rethrow_exception(failure_);
        if (closing_) throw std::runtime_error("Inference service is closing");
        queue_.push_back(request);
    }
    changed_.notify_all();
    std::unique_lock<std::mutex> lock(request->mutex);
    request->changed.wait(lock,[&]{return request->done;});
    if(request->error)std::rethrow_exception(request->error);
    return std::move(request->output);
}
void BatchEvaluator::serve(size_t index) {
    try { backends_[index]->initialize(); }
    catch (...) {
        { std::lock_guard<std::mutex> lock(mutex_);
          if (!failure_) failure_=std::current_exception();
          closing_=true;
          for(auto r:queue_)r->fail(failure_);
          queue_.clear(); }
        changed_.notify_all(); return;
    }
    std::vector<Request*> batch;batch.reserve(max_batch_);
    InferenceInputs inputs;inputs.reserve(max_batch_);
    for (;;) {
        batch.clear();inputs.clear();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            changed_.wait(lock, [&] { return !queue_.empty() || closing_; });
            if (queue_.empty()) return;
            // KataGo's default is take-up-to-N immediately, without a fill delay.
            if (wait_us_ > 0) {
                auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(wait_us_);
                changed_.wait_until(lock, deadline, [&] { return closing_ || queue_.size() >= max_batch_; });
            }
            while (!queue_.empty() && batch.size() < max_batch_) {
                batch.push_back(std::move(queue_.front())); queue_.pop_front();
            }
            ++active_;
        }
        changed_.notify_all();
        try {
            auto now = std::chrono::steady_clock::now();
            for (const auto& r : batch) {
                if(r->owner!=this || r->observation->size()!=input_size_)
                    throw std::runtime_error("Mixed inference contract/model in batch");
                wait_microseconds.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(now-r->submitted).count());
                inputs.push_back(r->observation);
            }
            auto outputs = backends_[index]->evaluate(inputs);
            if (outputs.size() != batch.size()) throw std::runtime_error("Inference output batch mismatch");
            // Validate the whole batch before fulfilling any promises, so failure wakes everyone once.
            for (const auto& out : outputs) {
                double mass=0;
                for(auto p:out.wdl) {
                    if(!std::isfinite(p)||p<0||p>1)throw std::runtime_error("Invalid backend WDL");
                    mass+=p;
                }
                if(std::abs(mass-1)>1e-5)throw std::runtime_error("Unnormalized backend WDL");
                if (out.logits.size() * INPUT_PLANES + GLOBAL_FEATURES != input_size_ || !std::isfinite(out.value()) || std::abs(out.value()) > 1.00001)
                    throw std::runtime_error("Invalid inference output value/shape");
                for (double p : out.logits) if (!std::isfinite(p)) throw std::runtime_error("Nonfinite policy logits");
            }
            requests.fetch_add(batch.size()); batches.fetch_add(1);
            rows_by_server[index]+=batch.size();
            uint64_t previous=max_observed_batch.load();
            while (previous<batch.size() && !max_observed_batch.compare_exchange_weak(previous,batch.size())) {}
            std::vector<std::shared_ptr<const Evaluation>> cached;
            if (!cache_.empty()) for (auto& out:outputs) cached.push_back(std::make_shared<const Evaluation>(out));
            for (size_t i = 0; i < batch.size(); ++i) {
                if (!cache_.empty()) {
                    auto& r=*batch[i];
                    CacheEntry replacement{r.cache_key,std::move(cached[i])};
                    {
                        std::lock_guard<std::mutex> lock(cache_locks_[r.cache_slot%cache_locks_.size()]);
                        std::swap(cache_[r.cache_slot],replacement);
                    } // Free the replaced key/output outside the stripe lock, as in KataGo.
                }
            }
            for(size_t i=0;i<batch.size();++i)batch[i]->complete(std::move(outputs[i]));
            {std::lock_guard<std::mutex> lock(mutex_);--active_;}
            changed_.notify_all();
        } catch (...) {
            auto error = std::current_exception();
            { std::lock_guard<std::mutex> lock(mutex_);
              if (!failure_) failure_ = error;
              closing_=true;
              --active_;
              for(auto r:batch)r->fail(error);
              for(auto r:queue_)r->fail(error);
              queue_.clear(); }
            changed_.notify_all();
            return;
        }
    }
}
}
