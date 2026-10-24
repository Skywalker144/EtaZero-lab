#pragma once
#include "neuralnet/inference.h"
#include <stdexcept>
class Rand;

namespace etazero::muzero {
struct Latent {
    virtual ~Latent() = default;
    virtual std::vector<float> copy_to_cpu() const { throw std::runtime_error("Latent has no diagnostic tensor"); }
};
struct InferenceOutput { std::shared_ptr<const Latent> latent; Evaluation evaluation; };
struct InferenceAction { std::shared_ptr<const Latent> latent; int action; };
class Backend {
protected:
    GameSpec spec_;
public:
    explicit Backend(GameSpec spec=game_spec(GameKind::GOMOKU)):spec_(spec) {}
    GameSpec spec() const { return spec_; }
    virtual ~Backend() = default;
    virtual void initialize() {} // Called serially, before any service threads start.
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
        uint64_t order=0;
    };
    struct CacheEntry { std::string key;InferenceOutput output; };
    std::vector<std::unique_ptr<Backend>> backends_;
    std::shared_ptr<const int> owner_=std::make_shared<const int>(0);
    std::atomic<uint64_t> next_request_{0};
    std::vector<std::thread> threads_;
    std::deque<Request*> initial_;
    std::vector<std::deque<Request*>> recurrent_;
    std::vector<CacheEntry> cache_;
    std::mutex mutex_,random_mutex_;std::deque<std::mutex> cache_locks_;
    std::condition_variable changed_;
    size_t batch_,capacity_,queued_=0,active_=0;int wait_us_,canvas_,symmetry_;bool randomize_;
    std::mt19937_64 random_;
    bool closing_=false;std::exception_ptr error_;
    void serve(size_t index);
    InferenceOutput submit(Request&,size_t server);
public:
    BatchEvaluator(std::vector<std::unique_ptr<Backend>>,int canvas,size_t batch,int wait_us,
                   size_t cache_entries,bool randomize,int symmetry,uint64_t seed,size_t queue_capacity=1024,int nn_mutex_pool_power_of_two=16);
    ~BatchEvaluator() override;
    InferenceOutput initial(const std::vector<float>&) override;
    InferenceOutput recurrent(std::shared_ptr<const Latent>,int) override;
    Evaluation evaluate(const std::vector<float>&) override;
    Evaluation evaluate_symmetry(const std::vector<float>&,int,bool=false,double=1,bool=false,double=0) override;
    void finish() override;
    void drain() override;
    void reset_stats() override;
    void clear_cache() override;
};
class RandomBackend final : public Backend {
    struct State : Latent { bool allow_no_result=false; std::vector<uint8_t> mask; };
    int canvas_;std::unique_ptr<Rand> random_;
    InferenceOutput output(std::shared_ptr<State>);
public:
    RandomBackend(int canvas,uint64_t seed,GameSpec spec=game_spec(GameKind::GOMOKU));
    ~RandomBackend() override;
    std::vector<InferenceOutput> initial(const InferenceInputs&) override;
    std::vector<InferenceOutput> recurrent(const std::vector<InferenceAction>&) override;
};
// Canonical -> fixed tree orientation. No latent rotation is performed.
std::vector<int> symmetry_mapping(int canvas,int symmetry);
std::vector<float> transform_observation(const std::vector<float>&,int canvas,const std::vector<int>&,GameSpec);
void restore_evaluation(Evaluation&,const std::vector<int>&);
}
