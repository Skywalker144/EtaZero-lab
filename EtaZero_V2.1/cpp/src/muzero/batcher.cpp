#include "etazero/muzero/inference.h"
#include "etazero/schema.h"
#include "etazero/symmetry.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace etazero::muzero {
std::vector<int> symmetry_mapping(int n,int s) {
    if(s<0 || s>7)throw std::runtime_error("Invalid MuZero symmetry");
    std::vector<int> map(n*n);
    for(int a=0;a<n*n;++a) {
        int x=a%n,y=a/n,tx=x,ty=y;
        switch(s){case 1:tx=y;ty=n-1-x;break;case 2:tx=n-1-x;ty=n-1-y;break;
        case 3:tx=n-1-y;ty=x;break;case 4:tx=y;ty=x;break;case 5:tx=n-1-x;break;
        case 6:tx=n-1-y;ty=n-1-x;break;case 7:ty=n-1-y;break;}
        map[a]=ty*n+tx;
    }
    return map;
}
std::vector<float> transform_observation(const std::vector<float>& obs,int n,const std::vector<int>& map) {
    if(obs.size()!=static_cast<size_t>(INPUT_PLANES*n*n+GLOBAL_FEATURES))throw std::runtime_error("MuZero observation shape mismatch");
    auto result=obs;
    for(int p=0;p<INPUT_PLANES;++p)for(int a=0;a<n*n;++a)result[p*n*n+map[a]]=obs[p*n*n+a];
    return result;
}
void restore_evaluation(Evaluation& e,const std::vector<int>& map) {
    if(e.logits.size()!=map.size())throw std::runtime_error("MuZero policy shape mismatch");
    auto logits=e.logits,optimistic=e.optimistic_logits;
    for(size_t a=0;a<map.size();++a){e.logits[a]=logits[map[a]];if(e.has_auxiliary)e.optimistic_logits[a]=optimistic.at(map[a]);}
}
BatchEvaluator::BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends,int canvas,size_t batch,int wait,
                               size_t cache,bool randomize,int symmetry,uint64_t seed,size_t capacity)
    :backends_(std::move(backends)),recurrent_(backends_.size()),cache_(cache),batch_(batch),capacity_(capacity),wait_us_(wait),
     canvas_(canvas),symmetry_(symmetry),randomize_(randomize),random_(seed) {
    if(backends_.empty() || !batch || !capacity || wait<0)throw std::runtime_error("Invalid MuZero service configuration");
    rows_by_server.resize(backends_.size());
    try{for(size_t i=0;i<backends_.size();++i)threads_.emplace_back(&BatchEvaluator::serve,this,i);}
    catch(...){finish();throw;}
}
BatchEvaluator::~BatchEvaluator(){try{finish();}catch(...){}}
InferenceOutput BatchEvaluator::submit(Request& request,size_t server) {
    request.done=false;request.error=nullptr;request.time=std::chrono::steady_clock::now();
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
    ++submitted;std::string key;size_t slot=0;
    if(!cache_.empty()) {
        // Input is already in the fixed tree orientation; exact globals retain PDA/rules.
        key.assign(reinterpret_cast<const char*>(obs.data()),obs.size()*sizeof(float));
        slot=std::hash<std::string>{}(key)%cache_.size();
        {std::lock_guard<std::mutex> lock(mutex_);if(error_)std::rethrow_exception(error_);if(closing_)throw std::runtime_error("MuZero service is closing");}
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if(cache_[slot].key==key && cache_[slot].output.latent){++cache_hits;return cache_[slot].output;}
    }
    thread_local Request request;request.observation=&obs;auto output=submit(request,0);
    if(!cache_.empty()){std::lock_guard<std::mutex> lock(cache_mutex_);cache_[slot]={std::move(key),output};}
    return output;
}
InferenceOutput BatchEvaluator::recurrent(std::shared_ptr<const Latent> latent,int action) {
    ++submitted;auto routed=std::dynamic_pointer_cast<const RoutedLatent>(latent);
    if(!routed || routed->owner!=owner_)throw std::runtime_error("MuZero latent belongs to another inference service");
    thread_local Request request;request.action={routed->value,action};return submit(request,routed->server);
}
Evaluation BatchEvaluator::evaluate(const std::vector<float>& obs) {
    return evaluate_symmetry(obs,symmetry_,false,1,randomize_);
}
Evaluation BatchEvaluator::evaluate_symmetry(const std::vector<float>& obs,int symmetry,bool skip,double temperature,bool randomize,double optimism) {
    if(!std::isfinite(temperature)||temperature<=0||!std::isfinite(optimism)||optimism<0||optimism>1)
        throw std::runtime_error("Invalid MuZero evaluation settings");
    input_symmetry(obs,symmetry);
    if(randomize){std::lock_guard<std::mutex> lock(random_mutex_);symmetry=std::uniform_int_distribution<int>(0,hex_input(obs)?1:7)(random_);}
    auto map=symmetry_mapping(canvas_,input_symmetry(obs,symmetry));
    auto input=transform_observation(obs,canvas_,map);InferenceOutput result;
    if(skip){++submitted;thread_local Request request;request.observation=&input;result=submit(request,0);}else result=initial(input);
    restore_evaluation(result.evaluation,map);return result.evaluation;
}
void BatchEvaluator::serve(size_t server) {
    auto complete=[](Request* r,InferenceOutput output,std::exception_ptr error) {
        std::lock_guard<std::mutex> lock(r->mutex);r->output=std::move(output);r->error=error;r->done=true;r->changed.notify_one();
    };
    std::vector<Request*> batch;
    InferenceInputs initial_inputs;std::vector<InferenceAction> recurrent_inputs;
    batch.reserve(batch_);initial_inputs.reserve(batch_);recurrent_inputs.reserve(batch_);
    try {
        backends_[server]->initialize();
        for(;;) {
            bool initial;
            {std::unique_lock<std::mutex> lock(mutex_);
                changed_.wait(lock,[&]{return closing_||!initial_.empty()||!recurrent_[server].empty();});
                if(error_)return;
                if(initial_.empty()&&recurrent_[server].empty()){if(closing_)return;continue;}
                if(wait_us_>0)changed_.wait_for(lock,std::chrono::microseconds(wait_us_),[&]{return closing_||initial_.size()>=batch_||recurrent_[server].size()>=batch_;});
                if(initial_.empty()&&recurrent_[server].empty())continue;
                initial=recurrent_[server].empty()||(!initial_.empty()&&initial_.front()->time<recurrent_[server].front()->time);
                auto& queue=initial?initial_:recurrent_[server];
                batch.clear();while(!queue.empty()&&batch.size()<batch_){batch.push_back(queue.front());queue.pop_front();}queued_-=batch.size();++active_;changed_.notify_all();
            }
            uint64_t waiting=0;
            for(auto r:batch)waiting+=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-r->time).count();
            wait_microseconds+=waiting;
            std::vector<InferenceOutput> outputs;
            if(initial){initial_inputs.clear();for(auto r:batch)initial_inputs.push_back(r->observation);outputs=backends_[server]->initial(initial_inputs);}
            else{recurrent_inputs.clear();for(auto r:batch)recurrent_inputs.push_back(r->action);outputs=backends_[server]->recurrent(recurrent_inputs);recurrent_inputs.clear();}
            if(outputs.size()!=batch.size())throw std::runtime_error("MuZero batch output count mismatch");
            for(auto& output:outputs) {
                if(!output.latent)throw std::runtime_error("Missing MuZero latent");
                auto routed=std::make_shared<RoutedLatent>();routed->value=output.latent;routed->owner=owner_;routed->server=server;output.latent=std::move(routed);
            }
            requests+=batch.size();++batches;rows_by_server[server]+=batch.size();
            auto maximum=max_observed_batch.load();while(maximum<batch.size()&&!max_observed_batch.compare_exchange_weak(maximum,batch.size())){}
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
void BatchEvaluator::reset_stats(){drain();requests=0;batches=0;max_observed_batch=0;wait_microseconds=0;submitted=0;cache_hits=0;std::fill(rows_by_server.begin(),rows_by_server.end(),0);}

namespace {uint64_t mix(uint64_t x){x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);}}
InferenceOutput RandomBackend::output(std::shared_ptr<State> state) {
    std::mt19937_64 rng(state->key);Evaluation e;e.logits.resize(canvas_*canvas_);
    for(auto& p:e.logits)p=std::uniform_real_distribution<double>(-1,1)(rng);
    double win=std::uniform_real_distribution<double>(0,1)(rng);e.wdl={win,0,1-win};return {state,std::move(e)};
}
std::vector<InferenceOutput> RandomBackend::initial(const InferenceInputs& inputs) {
    std::vector<InferenceOutput> result;
    for(auto input:inputs){if(!input||input->size()!=static_cast<size_t>(INPUT_PLANES*canvas_*canvas_+GLOBAL_FEATURES))throw std::runtime_error("Random MuZero input shape");
        auto state=std::make_shared<State>();state->key=seed_;state->mask.resize(canvas_*canvas_);
        for(size_t i=0;i<input->size();++i){uint32_t bits;std::memcpy(&bits,&(*input)[i],sizeof(bits));state->key=mix(state->key^bits);if(i<state->mask.size())state->mask[i]=(*input)[i]!=0;}
        result.push_back(output(state));}
    return result;
}
std::vector<InferenceOutput> RandomBackend::recurrent(const std::vector<InferenceAction>& inputs) {
    std::vector<InferenceOutput> result;
    for(const auto& input:inputs){auto parent=std::dynamic_pointer_cast<const State>(input.latent);
        if(!parent||input.action<0||input.action>=canvas_*canvas_||!parent->mask[input.action])throw std::runtime_error("Invalid random MuZero transition");
        auto state=std::make_shared<State>(*parent);state->key=mix(parent->key^mix(input.action));result.push_back(output(state));}
    return result;
}
}
