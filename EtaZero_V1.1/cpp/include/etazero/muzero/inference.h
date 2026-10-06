#pragma once
#include "etazero/inference.h"
#include <stdexcept>

namespace etazero::muzero {
struct Latent {
    virtual ~Latent() = default;
    virtual std::vector<float> copy_to_cpu() const { throw std::runtime_error("Latent has no diagnostic tensor"); }
};
struct InferenceOutput { std::shared_ptr<const Latent> latent; Evaluation evaluation; };
struct InferenceAction { std::shared_ptr<const Latent> latent; int action; };
class Backend {
public:
    virtual ~Backend() = default;
    virtual void initialize() {}
    virtual std::vector<InferenceOutput> initial(const InferenceInputs&) = 0;
    virtual std::vector<InferenceOutput> recurrent(const std::vector<InferenceAction>&) = 0;
};
class Evaluator : public InferenceService {
public:
    virtual InferenceOutput initial(const std::vector<float>& observation) = 0;
    virtual InferenceOutput recurrent(std::shared_ptr<const Latent>, int action) = 0;
};
class BatchEvaluator final : public Evaluator {
    struct RoutedLatent : Latent {
        std::shared_ptr<const Latent> value;
        std::shared_ptr<const int> owner;
        size_t server;
        std::vector<float> copy_to_cpu() const override { return value->copy_to_cpu(); }
    };
    struct Request {
        const std::vector<float>* observation=nullptr;
        InferenceAction action;
        InferenceOutput output;
        std::exception_ptr error;
        bool done=false;
        std::mutex mutex;std::condition_variable changed;
        std::chrono::steady_clock::time_point time=std::chrono::steady_clock::now();
    };
    struct CacheEntry { std::string key;InferenceOutput output; };
    std::vector<std::unique_ptr<Backend>> backends_;
    std::shared_ptr<const int> owner_=std::make_shared<const int>(0);
    std::vector<std::thread> threads_;
    std::deque<Request*> initial_;
    std::vector<std::deque<Request*>> recurrent_;
    std::vector<CacheEntry> cache_;
    std::mutex mutex_,cache_mutex_,random_mutex_;
    std::condition_variable changed_;
    size_t batch_,capacity_,queued_=0,active_=0;int wait_us_,canvas_,symmetry_;bool randomize_;
    std::mt19937_64 random_;
    bool closing_=false;std::exception_ptr error_;
    void serve(size_t index);
    InferenceOutput submit(Request&,size_t server);
public:
    BatchEvaluator(std::vector<std::unique_ptr<Backend>>,int canvas,size_t batch,int wait_us,
                   size_t cache_entries,bool randomize,int symmetry,uint64_t seed,size_t queue_capacity=1024);
    ~BatchEvaluator() override;
    InferenceOutput initial(const std::vector<float>&) override;
    InferenceOutput recurrent(std::shared_ptr<const Latent>,int) override;
    Evaluation evaluate(const std::vector<float>&) override;
    Evaluation evaluate_symmetry(const std::vector<float>&,int,bool=false,double=1,bool=false,double=0) override;
    void finish() override;
    void drain() override;
    void reset_stats() override;
};
class RandomBackend final : public Backend {
    struct State : Latent { uint64_t key; std::vector<uint8_t> mask; };
    int canvas_;uint64_t seed_;
    InferenceOutput output(std::shared_ptr<State>);
public:
    RandomBackend(int canvas,uint64_t seed):canvas_(canvas),seed_(seed){}
    std::vector<InferenceOutput> initial(const InferenceInputs&) override;
    std::vector<InferenceOutput> recurrent(const std::vector<InferenceAction>&) override;
};
// Canonical -> fixed tree orientation. No latent rotation is performed.
std::vector<int> symmetry_mapping(int canvas,int symmetry);
std::vector<float> transform_observation(const std::vector<float>&,int canvas,const std::vector<int>&);
void restore_evaluation(Evaluation&,const std::vector<int>&);
}
