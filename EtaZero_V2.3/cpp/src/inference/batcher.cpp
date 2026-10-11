#include "etazero/inference.h"
#include "etazero/schema.h"
#include "etazero/symmetry.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace etazero {
namespace {
std::vector<int> symmetry_mapping(size_t input_size,int symmetry) {
    if(symmetry<0 || symmetry>=8 || input_size<GLOBAL_FEATURES)throw std::runtime_error("Invalid D4 request");
    size_t area=(input_size-GLOBAL_FEATURES)/INPUT_PLANES;
    int n=static_cast<int>(std::sqrt(area));
    if(n*n!=static_cast<int>(area) || area*INPUT_PLANES+GLOBAL_FEATURES!=input_size)
        throw std::runtime_error("Invalid D4 input shape");
    std::vector<int> mapping(area);
    for(int a=0;a<static_cast<int>(area);++a) {
        int x=a%n,y=a/n,tx=x,ty=y;
        switch(symmetry) {
            case 1:tx=y;ty=n-1-x;break;
            case 2:tx=n-1-x;ty=n-1-y;break;
            case 3:tx=n-1-y;ty=x;break;
            case 4:tx=y;ty=x;break;
            case 5:tx=n-1-x;break;
            case 6:tx=n-1-y;ty=n-1-x;break;
            case 7:ty=n-1-y;break;
        }
        mapping[a]=ty*n+tx;
    }
    return mapping;
}
std::vector<float> transform_input(const std::vector<float>& obs,const std::vector<int>& mapping) {
    auto transformed=obs;size_t area=mapping.size();
    for(int p=0;p<INPUT_PLANES;++p)for(size_t a=0;a<area;++a)
        transformed[p*area+mapping[a]]=obs[p*area+a];
    return transformed;
}
void restore_output(Evaluation& output,const std::vector<int>& mapping) {
    if(output.logits.size()!=mapping.size())throw std::runtime_error("Invalid symmetry policy shape");
    auto logits=output.logits;
    for(size_t a=0;a<mapping.size();++a)output.logits[a]=logits[mapping[a]];
    if(output.has_auxiliary) {
        if(output.optimistic_logits.size()!=mapping.size())throw std::runtime_error("Invalid optimistic symmetry shape");
        auto optimistic=output.optimistic_logits;
        for(size_t a=0;a<mapping.size();++a)output.optimistic_logits[a]=optimistic[mapping[a]];
    }
}
std::vector<std::unique_ptr<Backend>> single(std::unique_ptr<Backend> backend) {
    std::vector<std::unique_ptr<Backend>> result; result.push_back(std::move(backend)); return result;
}
}
Evaluation Evaluator::evaluate_symmetry(const std::vector<float>& obs,int symmetry,bool skip_cache,double temperature,bool randomize,double optimism) {
    (void)skip_cache;(void)temperature;(void)randomize;(void)optimism; // Generic evaluators use the caller's seeded fallback orientation.
    auto mapping=symmetry_mapping(obs.size(),input_symmetry(obs,symmetry));
    auto transformed=transform_input(obs,mapping);auto output=evaluate(transformed);
    restore_output(output,mapping);return output;
}
BatchEvaluator::BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                               size_t max_batch, size_t capacity, int wait_us)
    : BatchEvaluator(single(std::move(backend)), std::move(model), canvas, max_batch, capacity, wait_us, 0) {}
BatchEvaluator::BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends, std::string model, int canvas,
                               size_t max_batch, size_t capacity, int wait_us, size_t cache_entries,bool randomize,int symmetry,uint64_t seed)
    : backends_(std::move(backends)), cache_(cache_entries), model_(std::move(model)),
      randomize_(randomize),symmetry_(symmetry),random_(seed),max_batch_(max_batch),
      capacity_(capacity), input_size_(INPUT_PLANES * canvas * canvas + GLOBAL_FEATURES), wait_us_(wait_us) {
    if (backends_.empty() || model_.empty() || !max_batch || !capacity || wait_us < 0 || symmetry<0 || symmetry>=8)
        throw std::runtime_error("Invalid inference service configuration");
    for (auto& b : backends_) if (!b) throw std::runtime_error("Null inference backend");
    for(const auto& backend:backends_)if(backend->supports_auxiliary()!=backends_[0]->supports_auxiliary())
        throw std::runtime_error("Inference servers disagree on auxiliary capability");
    rows_by_server.resize(backends_.size(),0);
    for(int s=0;s<8;++s)mappings_[s]=symmetry_mapping(input_size_,s);
    // CUDA graph capture must finish before another service can dispatch CUDA work.
    for(auto& backend:backends_)backend->initialize();
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
    return evaluate_symmetry(obs,symmetry_,false,1,randomize_);
}
Evaluation BatchEvaluator::evaluate_symmetry(const std::vector<float>& obs,int symmetry,bool skip_cache,double temperature,bool randomize,double optimism) {
    if (obs.size() != input_size_) throw std::runtime_error("Inference input shape mismatch");
    if(!std::isfinite(temperature) || temperature<=0)throw std::runtime_error("Invalid NN policy temperature");
    input_symmetry(obs,symmetry);
    if(!std::isfinite(optimism) || optimism<0 || optimism>1)throw std::runtime_error("Invalid policy optimism");
    // Unsupported backends have no optimistic output; match the source's absent-head condition.
    if(!backends_[0]->supports_auxiliary())optimism=0;
    // evaluate() is synchronous: the caller owns its input until completion.
    // One thread-local request replaces per-query shared_ptr/promise allocations.
    thread_local Request local;
    Request* request=&local;
    ++submitted;
    request->use_cache=!skip_cache && !cache_.empty();request->symmetry=symmetry;
    if (request->use_cache) {
        // Pack binary spatial planes and append exact global float bytes. The
        // evaluator is bound to one model, precision, and canvas.
        const size_t spatial = obs.size() - GLOBAL_FEATURES;
        request->cache_key.assign((spatial+7)/8, '\0');
        for (size_t i=0;i<spatial;++i) {
            if (obs[i]!=0 && obs[i]!=1) throw std::runtime_error("Cache requires binary input features");
            if (obs[i]) request->cache_key[i/8] |= static_cast<char>(1 << (7-i%8));
        }
        request->cache_key.append(reinterpret_cast<const char*>(obs.data()+spatial), GLOBAL_FEATURES*sizeof(float));
        // Orientation is excluded, as in KataGo's NNInputParams hash. NN temperature
        // is included even though this adapter stores canonical raw logits.
        request->cache_key.append(reinterpret_cast<const char*>(&temperature),sizeof(temperature));
        request->cache_key.append(reinterpret_cast<const char*>(&optimism),sizeof(optimism));
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
    if(randomize) {
        std::lock_guard<std::mutex> lock(random_mutex_);
        symmetry=std::uniform_int_distribution<int>(0,hex_input(obs)?1:7)(random_);
    }
    request->symmetry=input_symmetry(obs,symmetry);
    const auto& mapping=mappings_[request->symmetry];
    auto& transformed=request->transformed;transformed.resize(obs.size());
    const size_t area=mapping.size();
    for(int p=0;p<INPUT_PLANES;++p)for(size_t a=0;a<area;++a)
        transformed[p*area+mapping[a]]=obs[p*area+a];
    std::copy(obs.begin()+INPUT_PLANES*area,obs.end(),transformed.begin()+INPUT_PLANES*area);
    static std::atomic<uint64_t> next{0};
    request->id=next.fetch_add(1);request->owner=this;request->observation=&transformed;
    request->submitted = std::chrono::steady_clock::now();
    {std::lock_guard<std::mutex> lock(request->mutex);request->done=false;request->error=nullptr;}
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // KataGo forcePush accepts pending queries beyond the nominal queue
        // capacity. Only the result wait blocks the caller, not queue admission.
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
            // Another server may consume the queue while this timed wait
            // releases the mutex. Never dispatch an empty batch to a backend.
            if (queue_.empty()) continue;
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
            // Layout belongs to the batch boundary; search validates numerical
            // values while consuming policies/WDL, without a duplicate scan.
            for (const auto& out : outputs) {
                if (out.logits.size() * INPUT_PLANES + GLOBAL_FEATURES != input_size_)
                    throw std::runtime_error("Invalid inference output shape");
                if(out.has_auxiliary && out.optimistic_logits.size()!=out.logits.size())
                    throw std::runtime_error("Invalid auxiliary inference output shape");
            }
            for(const auto& output:outputs)if(output.has_auxiliary!=backends_[index]->supports_auxiliary())
                throw std::runtime_error("Backend auxiliary output violates capability contract");
            for(size_t i=0;i<outputs.size();++i)
                restore_output(outputs[i],mappings_[batch[i]->symmetry]);
            requests.fetch_add(batch.size()); batches.fetch_add(1);
            rows_by_server[index]+=batch.size();
            uint64_t previous=max_observed_batch.load();
            while (previous<batch.size() && !max_observed_batch.compare_exchange_weak(previous,batch.size())) {}
            for (size_t i = 0; i < batch.size(); ++i) {
                if (batch[i]->use_cache) {
                    auto& r=*batch[i];
                    CacheEntry replacement{r.cache_key,std::make_shared<const Evaluation>(outputs[i])};
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
