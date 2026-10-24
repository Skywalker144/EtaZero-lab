#include "etazero/muzero/inference.h"
#include "etazero/random_evaluator.h"
#include "../../external/katago/core/rand.h"
#include "etazero/schema.h"
#include "etazero/symmetry.h"
#include <algorithm>
#include <cmath>

namespace etazero::muzero {
std::vector<int> symmetry_mapping(int n,int s) {
    if(s<0 || s>11)throw std::runtime_error("Invalid MuZero symmetry");
    return spatial_mapping(n,s);
}
std::vector<float> transform_observation(const std::vector<float>& obs,int n,const std::vector<int>& map,GameSpec spec) {
    if(obs.size()!=static_cast<size_t>(spec.planes*n*n+spec.globals))throw std::runtime_error("MuZero observation shape mismatch");
    auto result=obs;
    for(int p=0;p<spec.planes;++p)for(int a=0;a<n*n;++a)result[p*n*n+map[a]]=obs[p*n*n+a];
    return result;
}
void restore_evaluation(Evaluation& e,const std::vector<int>& map) {
    if(e.logits.size()!=map.size())throw std::runtime_error("MuZero policy shape mismatch");
    thread_local std::vector<double> scratch;
    scratch=e.logits;
    for(size_t a=0;a<map.size();++a)e.logits[a]=scratch[map[a]];
    if(e.has_auxiliary) {
        if(e.optimistic_logits.size()!=map.size())throw std::runtime_error("MuZero optimistic policy shape mismatch");
        scratch=e.optimistic_logits;
        for(size_t a=0;a<map.size();++a)e.optimistic_logits[a]=scratch[map[a]];
    }
}
BatchEvaluator::BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends,int canvas,size_t batch,int wait,
                               size_t cache,bool randomize,int symmetry,uint64_t seed,size_t capacity,int mutex_power)
    :backends_(std::move(backends)),recurrent_(backends_.size()),cache_(cache),batch_(batch),capacity_(capacity),wait_us_(wait),
     canvas_(canvas),symmetry_(symmetry),randomize_(randomize),random_(seed) {
    if(backends_.empty() || !batch || !capacity || wait<0)throw std::runtime_error("Invalid MuZero service configuration");
    if(mutex_power< -1||mutex_power>24)throw std::runtime_error("Invalid MuZero cache mutex pool power");
    if(mutex_power<0 && !cache_.empty())throw std::runtime_error("Enabled NN cache requires a nonnegative mutex pool power");
    cache_locks_.resize(std::min(cache_.size(),size_t(1)<<std::max(0,mutex_power)));
    for(auto& backend:backends_)backend->initialize();
    spec_=backends_[0]->spec();
    for(const auto& backend:backends_)if(!same_layout(spec_,backend->spec()))throw std::runtime_error("MuZero server layout mismatch");
    rows_by_server.resize(backends_.size());
    try{for(size_t i=0;i<backends_.size();++i)threads_.emplace_back(&BatchEvaluator::serve,this,i);}
    catch(...){finish();throw;}
}
BatchEvaluator::~BatchEvaluator(){try{finish();}catch(...){}}
InferenceOutput BatchEvaluator::submit(Request& request,size_t server) {
    request.done=false;request.error=nullptr;
    // Preserve oldest-request scheduling without reading the wall clock.
    request.order=next_request_.fetch_add(1,std::memory_order_relaxed);
    {std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock,[&]{return error_||closing_||queued_<capacity_;});
        if(error_)std::rethrow_exception(error_);
        if(closing_)throw std::runtime_error("MuZero service is closing");
        (request.observation?initial_:recurrent_.at(server)).push_back(&request);++queued_;
    }
    changed_.notify_all();std::unique_lock<std::mutex> lock(request.mutex);
    request.changed.wait(lock,[&]{return request.done;});
    if(request.error)std::rethrow_exception(request.error);
    auto output=std::move(request.output);
    request.observation=nullptr;request.action={};
    return output;
}
InferenceOutput BatchEvaluator::initial(const std::vector<float>& obs) {
    auto spec=spec_;
    if(obs.size()!=static_cast<size_t>(spec.planes*canvas_*canvas_+spec.globals))
        throw std::runtime_error("MuZero observation shape mismatch");
    submitted.fetch_add(1,std::memory_order_relaxed);std::string key;size_t slot=0;
    if(!cache_.empty()) {
        // Input is already in the fixed tree orientation; exact globals retain PDA/rules.
        key.assign(reinterpret_cast<const char*>(obs.data()),obs.size()*sizeof(float));
        slot=std::hash<std::string>{}(key)%cache_.size();
        {std::lock_guard<std::mutex> lock(mutex_);if(error_)std::rethrow_exception(error_);if(closing_)throw std::runtime_error("MuZero service is closing");}
        std::lock_guard<std::mutex> lock(cache_locks_[slot%cache_locks_.size()]);
        if(cache_[slot].key==key && cache_[slot].output.latent){cache_hits.fetch_add(1,std::memory_order_relaxed);return cache_[slot].output;}
    }
    thread_local Request request;request.observation=&obs;auto output=submit(request,0);
    if(!cache_.empty()){std::lock_guard<std::mutex> lock(cache_locks_[slot%cache_locks_.size()]);cache_[slot]={std::move(key),output};}
    return output;
}
InferenceOutput BatchEvaluator::recurrent(std::shared_ptr<const Latent> latent,int action) {
    submitted.fetch_add(1,std::memory_order_relaxed);auto routed=std::dynamic_pointer_cast<const RoutedLatent>(latent);
    if(!routed || routed->owner!=owner_)throw std::runtime_error("MuZero latent belongs to another inference service");
    thread_local Request request;request.action={routed->value,action};return submit(request,routed->server);
}
Evaluation BatchEvaluator::evaluate(const std::vector<float>& obs) {
    return evaluate_symmetry(obs,symmetry_,false,1,randomize_);
}
Evaluation BatchEvaluator::evaluate_symmetry(const std::vector<float>& obs,int symmetry,bool skip,double temperature,bool randomize,double optimism) {
    if(!std::isfinite(temperature)||temperature<=0||!std::isfinite(optimism)||optimism<0||optimism>1)
        throw std::runtime_error("Invalid MuZero evaluation settings");
    input_symmetry(obs,spec_,symmetry);
    if(randomize){std::lock_guard<std::mutex> lock(random_mutex_);symmetry=std::uniform_int_distribution<int>(0,symmetry_count(obs,spec_)-1)(random_);}
    auto map=symmetry_mapping(canvas_,input_symmetry(obs,spec_,symmetry));
    auto input=transform_observation(obs,canvas_,map,spec_);InferenceOutput result;
    if(skip){submitted.fetch_add(1,std::memory_order_relaxed);thread_local Request request;request.observation=&input;result=submit(request,0);}else result=initial(input);
    restore_evaluation(result.evaluation,policy_mapping(map,spec_));return result.evaluation;
}
void BatchEvaluator::serve(size_t server) {
    auto complete=[](Request* r,InferenceOutput output,std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(r->mutex);r->output=std::move(output);r->error=error;r->done=true;r->changed.notify_one();
    };
    std::vector<Request*> batch;
    InferenceInputs initial_inputs;std::vector<InferenceAction> recurrent_inputs;
    batch.reserve(batch_);initial_inputs.reserve(batch_);recurrent_inputs.reserve(batch_);
    try {
        for(;;) {
            bool initial;
            {std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&]{return closing_||!initial_.empty()||!recurrent_[server].empty();});
                if(error_)return;
                if(initial_.empty()&&recurrent_[server].empty()){if(closing_)return;continue;}
                if(wait_us_>0)changed_.wait_for(lock,std::chrono::microseconds(wait_us_),[&]{return closing_||initial_.size()>=batch_||recurrent_[server].size()>=batch_;});
                if(initial_.empty()&&recurrent_[server].empty())continue;
                initial=recurrent_[server].empty()||(!initial_.empty()&&initial_.front()->order<recurrent_[server].front()->order);
                auto& queue=initial?initial_:recurrent_[server];
                batch.clear();while(!queue.empty()&&batch.size()<batch_){batch.push_back(queue.front());queue.pop_front();}queued_-=batch.size();++active_;changed_.notify_all();
            }
            std::vector<InferenceOutput> outputs;
            if(initial){initial_inputs.clear();for(auto r:batch)initial_inputs.push_back(r->observation);outputs=backends_[server]->initial(initial_inputs);}
            else{recurrent_inputs.clear();for(auto r:batch)recurrent_inputs.push_back(r->action);outputs=backends_[server]->recurrent(recurrent_inputs);recurrent_inputs.clear();}
            if(outputs.size()!=batch.size())throw std::runtime_error("MuZero batch output count mismatch");
            for(auto& output:outputs) {
                if(!output.latent)throw std::runtime_error("Missing MuZero latent");
                auto routed=std::make_shared<RoutedLatent>();routed->value=output.latent;routed->owner=owner_;routed->server=server;output.latent=std::move(routed);
            }
            requests.fetch_add(batch.size(),std::memory_order_relaxed);batches.fetch_add(1,std::memory_order_relaxed);rows_by_server[server]+=batch.size();
            auto maximum=max_observed_batch.load(std::memory_order_relaxed);while(maximum<batch.size()&&!max_observed_batch.compare_exchange_weak(maximum,batch.size(),std::memory_order_relaxed)){}
            for(size_t i=0;i<batch.size();++i){complete(batch[i],std::move(outputs[i]),nullptr);}
            {std::lock_guard<std::mutex> lock(mutex_);--active_;}batch.clear();changed_.notify_all();
        }
    } catch(...) {
        std::lock_guard<std::mutex> lock(mutex_);if(!error_)error_=std::current_exception();closing_=true;
        if(!batch.empty())--active_;
        for(auto r:batch)complete(r,{},error_);
        for(auto r:initial_)complete(r,{},error_);initial_.clear();
        for(auto& queue:recurrent_){for(auto r:queue)complete(r,{},error_);queue.clear();}
        queued_=0;changed_.notify_all();
    }
}
void BatchEvaluator::drain(){std::unique_lock<std::mutex> lock(mutex_);changed_.wait(lock,[&]{return error_||(active_==0&&initial_.empty()&&std::all_of(recurrent_.begin(),recurrent_.end(),[](const auto& q){return q.empty();}));});if(error_)std::rethrow_exception(error_);}
void BatchEvaluator::finish(){{std::lock_guard<std::mutex> lock(mutex_);closing_=true;}changed_.notify_all();for(auto& t:threads_)if(t.joinable())t.join();if(error_)std::rethrow_exception(error_);}
void BatchEvaluator::reset_stats(){drain();requests=0;batches=0;max_observed_batch=0;submitted=0;cache_hits=0;std::fill(rows_by_server.begin(),rows_by_server.end(),0);}

RandomBackend::RandomBackend(int canvas,uint64_t seed,GameSpec spec)
    :Backend(spec),canvas_(canvas),random_(std::make_unique<Rand>(seed)) {}
RandomBackend::~RandomBackend()=default;
InferenceOutput RandomBackend::output(std::shared_ptr<State> state) {
    return {state,random_evaluation(*random_,state->mask,state->allow_no_result)};
}
std::vector<InferenceOutput> RandomBackend::initial(const InferenceInputs& inputs) {
    std::vector<InferenceOutput> result;
    for(auto input:inputs){if(!input)throw std::runtime_error("Random MuZero null input");
        auto spec=spec_;
        if(input->size()!=static_cast<size_t>(spec.planes*canvas_*canvas_+spec.globals))throw std::runtime_error("Random MuZero observation shape");
        auto state=std::make_shared<State>();state->mask.resize(spec.actions(canvas_),1);
        state->allow_no_result=no_result_possible(*input,spec,canvas_);
        if(spec.action_layout!=ActionLayout::MOVES)
            for(int i=0;i<canvas_*canvas_;++i)state->mask[i]=(*input)[i]!=0;
        result.push_back(output(state));}
    return result;
}
std::vector<InferenceOutput> RandomBackend::recurrent(const std::vector<InferenceAction>& inputs) {
    std::vector<InferenceOutput> result;
    for(const auto& input:inputs){auto parent=std::dynamic_pointer_cast<const State>(input.latent);
        if(!parent||input.action<0||input.action>=static_cast<int>(parent->mask.size())||!parent->mask[input.action])throw std::runtime_error("Invalid random MuZero transition");
        result.push_back(output(std::make_shared<State>(*parent)));}
    return result;
}
}

namespace etazero::muzero {
void BatchEvaluator::clear_cache() {
    for(size_t i=0;i<cache_.size();++i){std::lock_guard<std::mutex> lock(cache_locks_[i%cache_locks_.size()]);cache_[i]={};}
    if(auto* service=dynamic_cast<InferenceService*>(secondary.get()))service->clear_cache();
}
}
