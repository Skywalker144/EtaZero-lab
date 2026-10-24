#include "neuralnet/inference.h"
#include "etazero/schema.h"
#include "neuralnet/sgfmetadata.h"
#include "game/symmetry.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace etazero {
ProfileEvaluator::ProfileEvaluator(Evaluator& d,const std::string& profile):delegate_(d){spec_=d.spec();d.validate_human_profile(profile);profile_=std::make_shared<SGFMetadata>(SGFMetadata::getProfile(profile));}
bool ProfileEvaluator::available() const {return profile_->initialized;}
bool no_result_possible(const std::vector<float>& observation,GameSpec spec,int canvas) {
    if(spec.kind!=GameKind::GO)return false;
    size_t global=spec.planes*canvas*canvas;
    return observation.at(global+6)==0 || observation.at(global+9)!=0;
}
WDL outcome_probabilities(WDL logits,bool allow_no_result) {
    if(!allow_no_result)logits[3]=-std::numeric_limits<double>::infinity();
    double maximum=*std::max_element(logits.begin(),logits.end()),sum=0;
    for(auto& p:logits){p=std::exp(p-maximum);sum+=p;}
    if(!std::isfinite(sum)||sum<=0)throw std::runtime_error("Invalid outcome logits");
    for(auto& p:logits)p=static_cast<float>(p/sum);
    return logits;
}
namespace {
std::vector<float> transform_input(const std::vector<float>& obs,const std::vector<int>& mapping,GameSpec spec) {
    auto transformed=obs;size_t area=mapping.size();
    for(int p=0;p<spec.planes;++p)for(size_t a=0;a<area;++a)
        transformed[p*area+mapping[a]]=obs[p*area+a];
    return transformed;
}
void restore_output(Evaluation& output,const std::vector<int>& spatial,GameSpec spec) {
    auto extended=spec.action_layout==ActionLayout::POINTS?std::vector<int>{}:policy_mapping(spatial,spec);
    const auto& mapping=extended.empty()?spatial:extended;
    if(output.logits.size()!=mapping.size())throw std::runtime_error("Invalid symmetry policy shape");
    thread_local std::vector<double> scratch;
    scratch=output.logits;
    for(size_t a=0;a<mapping.size();++a)output.logits[a]=scratch[mapping[a]];
    if(output.has_auxiliary) {
        if(output.optimistic_logits.size()!=mapping.size())throw std::runtime_error("Invalid optimistic symmetry shape");
        scratch=output.optimistic_logits;
        for(size_t a=0;a<mapping.size();++a)output.optimistic_logits[a]=scratch[mapping[a]];
    }
}
std::vector<std::unique_ptr<Backend>> single(std::unique_ptr<Backend> backend) {
    std::vector<std::unique_ptr<Backend>> result; result.push_back(std::move(backend)); return result;
}
}
Evaluation Evaluator::evaluate_symmetry(const std::vector<float>& obs,int symmetry,bool skip_cache,double temperature,bool randomize,double optimism) {
    (void)skip_cache;(void)temperature;(void)randomize;(void)optimism; // Generic evaluators use the caller's seeded fallback orientation.
    if(obs.size()<static_cast<size_t>(spec_.globals))throw std::runtime_error("Invalid observation shape");
    int canvas=static_cast<int>(std::sqrt((obs.size()-spec_.globals)/spec_.planes));
    if(obs.size()!=static_cast<size_t>(spec_.planes*canvas*canvas+spec_.globals))throw std::runtime_error("Invalid observation shape");
    auto mapping=spatial_mapping(canvas,input_symmetry(obs,spec_,symmetry));
    auto transformed=transform_input(obs,mapping,spec_);auto output=evaluate(transformed);
    restore_output(output,mapping,spec_);return output;
}
BatchEvaluator::BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                               size_t max_batch, int wait_us)
    : BatchEvaluator(single(std::move(backend)), std::move(model), canvas, max_batch, wait_us, 0) {}
BatchEvaluator::BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends, std::string model, int canvas,
                               size_t max_batch, int wait_us, size_t cache_entries,bool randomize,int symmetry,uint64_t seed,int mutex_pool_power_of_two)
    : backends_(std::move(backends)), cache_(cache_entries), model_(std::move(model)),
      randomize_(randomize),symmetry_(symmetry),random_(seed),max_batch_(max_batch),
      input_size_(0), wait_us_(wait_us),canvas_(canvas) {
    if (backends_.empty() || model_.empty() || !max_batch || wait_us < 0 || symmetry<0 || symmetry>=8)
        throw std::runtime_error("Invalid inference service configuration");
    if(mutex_pool_power_of_two< -1 || mutex_pool_power_of_two>24)throw std::runtime_error("Invalid NN cache mutex pool power");
    if(mutex_pool_power_of_two<0 && !cache_.empty())throw std::runtime_error("Enabled NN cache requires a nonnegative mutex pool power");
    cache_locks_.resize(std::max<size_t>(1,std::min(cache_.size(),size_t(1)<<std::max(0,mutex_pool_power_of_two))));
    for (auto& b : backends_) if (!b) throw std::runtime_error("Null inference backend");
    for(const auto& backend:backends_)if(backend->supports_auxiliary()!=backends_[0]->supports_auxiliary())
        throw std::runtime_error("Inference servers disagree on auxiliary capability");
    rows_by_server.resize(backends_.size(),0);
    // CUDA graph capture must finish before another service can dispatch CUDA work.
    for(auto& backend:backends_)backend->initialize();
    spec_=backends_[0]->spec();input_size_=spec_.planes*canvas*canvas+spec_.globals+backends_[0]->metadata_channels();
    for(const auto& backend:backends_)if(!same_layout(spec_,backend->spec()))throw std::runtime_error("Inference server layout mismatch");
    for(int s=0;s<(spec_.symmetry==SymmetryKind::XIANGQI?12:8);++s)mappings_[s]=spatial_mapping(canvas,s);
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
    drain();requests=0;batches=0;max_observed_batch=0;cache_hits=0;submitted=0;
    std::fill(rows_by_server.begin(),rows_by_server.end(),0);
}
Evaluation BatchEvaluator::evaluate_position(const std::vector<float>& obs,int player,int symmetry,bool skip,double temperature,bool randomize,double optimism) {
    if(!backends_[0]->metadata_channels())return evaluate_symmetry(obs,symmetry,skip,temperature,randomize,optimism);
    auto input=obs;backends_[0]->append_metadata(input,player);
    return evaluate_symmetry(input,symmetry,skip,temperature,randomize,optimism);
}
Evaluation BatchEvaluator::evaluate(const std::vector<float>& obs) {
    return evaluate_symmetry(obs,symmetry_,false,1,randomize_);
}
Evaluation BatchEvaluator::evaluate_symmetry(const std::vector<float>& obs,int symmetry,bool skip_cache,double temperature,bool randomize,double optimism) {
    const auto spec=spec_;
    if(obs.size()!=input_size_)
        throw std::runtime_error("Inference input canvas mismatch");
    if(!std::isfinite(temperature) || temperature<=0)throw std::runtime_error("Invalid NN policy temperature");
    std::vector<float> base;
    const std::vector<float>* features=&obs;
    if(backends_[0]->metadata_channels()) {base.assign(obs.begin(),obs.end()-backends_[0]->metadata_channels());features=&base;}
    input_symmetry(*features,spec_,symmetry);
    if(!std::isfinite(optimism) || optimism<0 || optimism>1)throw std::runtime_error("Invalid policy optimism");
    // Unsupported backends have no optimistic output; match the source's absent-head condition.
    if(!backends_[0]->supports_auxiliary() || backends_[0]->metadata_channels()>0)optimism=0;
    // evaluate() is synchronous: the caller owns its input until completion.
    // One thread-local request replaces per-query shared_ptr/promise allocations.
    thread_local Request local;
    Request* request=&local;
    submitted.fetch_add(1,std::memory_order_relaxed);
    request->use_cache=!skip_cache && !cache_.empty();request->symmetry=symmetry;
    if (request->use_cache) {
        // Pack binary spatial planes and append exact global float bytes. The
        // evaluator is bound to one model, precision, and canvas.
        const size_t spatial = spec.planes*canvas_*canvas_;
        request->cache_key.assign((spatial+7)/8, '\0');
        for (size_t i=0;i<spatial;++i) {
            if (obs[i]) request->cache_key[i/8] |= static_cast<char>(1 << (7-i%8));
        }
        request->cache_key.append(reinterpret_cast<const char*>(obs.data()+spatial), (obs.size()-spatial)*sizeof(float));
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
            cache_hits.fetch_add(1,std::memory_order_relaxed);return *found;
        }
    }
    if(randomize) {
        std::lock_guard<std::mutex> lock(random_mutex_);
        symmetry=std::uniform_int_distribution<int>(0,symmetry_count(*features,spec_)-1)(random_);
    }
    request->symmetry=input_symmetry(*features,spec_,symmetry);
    const auto& mapping=mappings_[request->symmetry];
    auto& transformed=request->transformed;transformed.resize(obs.size());
    const size_t area=mapping.size();
    for(int p=0;p<spec.planes;++p)for(size_t a=0;a<area;++a)
        transformed[p*area+mapping[a]]=obs[p*area+a];
    std::copy(obs.begin()+spec.planes*area,obs.end(),transformed.begin()+spec.planes*area);
    request->owner=this;request->observation=&transformed;
    {std::lock_guard<std::mutex> lock(request->mutex);request->done=false;request->error=nullptr;}
    {
        std::unique_lock<std::mutex> lock(mutex_);
        // Like KataGo forcePush, only the result wait blocks the caller.
        // Outstanding synchronous callers bound the number of queued requests.
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
            for (const auto& r : batch) {
                if(r->owner!=this)
                    throw std::runtime_error("Mixed inference contract/model in batch");
                inputs.push_back(r->observation);
            }
            auto outputs = backends_[index]->evaluate(inputs);
            if (outputs.size() != batch.size()) throw std::runtime_error("Inference output batch mismatch");
            // Layout belongs to the batch boundary; search validates numerical
            // values while consuming policies/WDL, without a duplicate scan.
            for (const auto& out:outputs) {
                if (out.logits.size() != static_cast<size_t>(spec_.actions(canvas_)))
                    throw std::runtime_error("Invalid inference output shape");
                if(out.has_auxiliary && out.optimistic_logits.size()!=out.logits.size())
                    throw std::runtime_error("Invalid auxiliary inference output shape");
            }
            for(auto& output:outputs)output.policy_optimism_allowed=backends_[index]->metadata_channels()==0;
            for(const auto& output:outputs)if(output.has_auxiliary!=backends_[index]->supports_auxiliary())
                throw std::runtime_error("Backend auxiliary output violates capability contract");
            for(size_t i=0;i<outputs.size();++i)
                restore_output(outputs[i],mappings_[batch[i]->symmetry],spec_);
            requests.fetch_add(batch.size(),std::memory_order_relaxed); batches.fetch_add(1,std::memory_order_relaxed);
            rows_by_server[index]+=batch.size();
            uint64_t previous=max_observed_batch.load(std::memory_order_relaxed);
            while (previous<batch.size() && !max_observed_batch.compare_exchange_weak(previous,batch.size(),std::memory_order_relaxed)) {}
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

namespace etazero {

void BatchEvaluator::clear_cache() {
    for(size_t i=0;i<cache_.size();++i){std::lock_guard<std::mutex> lock(cache_locks_[i%cache_locks_.size()]);cache_[i]={};}
    if(auto* service=dynamic_cast<InferenceService*>(secondary.get()))service->clear_cache();
}
}

namespace etazero {
 void BatchEvaluator::set_human_profile(const std::string& value) {drain();for(auto& b:backends_)b->set_human_profile(value);clear_cache();} }

namespace etazero {

Evaluation BatchEvaluator::evaluate_profile(const std::vector<float>& obs,int player,int symmetry,bool skip,double temperature,bool randomize,double optimism,const SGFMetadata& profile) {
    auto input=obs;backends_[0]->append_profile(input,player,profile);return evaluate_symmetry(input,symmetry,skip,temperature,randomize,optimism);
}
}
