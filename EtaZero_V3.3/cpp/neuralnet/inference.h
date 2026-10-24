#pragma once
#include "game/game_spec.h"
#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <random>
struct SGFMetadata;
namespace etazero {
// Probabilities are W/D/L/no-result from the current player's perspective.
using WDL = std::array<double,4>;
struct ModelDescription { std::string name; int version=-1; bool uses_human_profile=false; bool prefer_pass_alive=true,prefer_exclude_atari=true; };
struct Evaluation {
    std::vector<double> logits;
    WDL wdl{0,1,0};
    std::vector<double> optimistic_logits;
    double shortterm_value_stdev=0;
    bool has_auxiliary=false;
    bool policy_optimism_allowed=true;
    // Keep backend logits unscaled until mixing and NN temperature are applied.
    float policy_output_scale=1;
    double variance_time=-1;
    std::vector<double> scaled_policy_logits(bool optimistic=false) const {
        auto result=optimistic?optimistic_logits:logits;
        for(auto& value:result)value=static_cast<float>(value)*policy_output_scale;
        return result;
    }
    double value() const { return wdl[0]-wdl[2]; }
};
// V7 globals encode ko at 6 and territory scoring at 9.
bool no_result_possible(const std::vector<float>& observation,GameSpec spec,int canvas);
WDL outcome_probabilities(WDL logits,bool allow_no_result);
using InferenceInputs = std::vector<const std::vector<float>*>;
class Evaluator {
protected:
    GameSpec spec_=game_spec(GameKind::GOMOKU);
public:
    std::shared_ptr<Evaluator> secondary;
    GameSpec spec() const { return spec_; }
    virtual ~Evaluator() = default;
    virtual bool available() const {return true;}
    virtual void validate_human_profile(const std::string&) const {throw std::runtime_error("Model has no human profile");}
    virtual Evaluation evaluate_profile(const std::vector<float>&,int,int,bool,double,bool,double,const SGFMetadata&) {throw std::runtime_error("Model has no human profile");}
    virtual Evaluation evaluate(const std::vector<float>& observation) = 0;
    virtual Evaluation evaluate_position(const std::vector<float>& observation, int player, int symmetry=0,
                                         bool skip_cache=false, double temperature=1, bool randomize=false, double optimism=0) {
        return evaluate_symmetry(observation,symmetry,skip_cache,temperature,randomize,optimism);
    }
    // Input and output use canonical board coordinates; symmetry is an NN request option.
    virtual Evaluation evaluate_symmetry(const std::vector<float>& observation, int symmetry,
                                         bool skip_cache=false, double policy_temperature=1, bool randomize=false, double policy_optimism=0);
};
class ProfileEvaluator final:public Evaluator {
    Evaluator& delegate_;std::shared_ptr<const SGFMetadata> profile_;
public:
    ProfileEvaluator(Evaluator& d,const std::string& profile);
    bool available() const override;
    Evaluation evaluate(const std::vector<float>&) override {throw std::runtime_error("Human metadata needs player perspective");}
    Evaluation evaluate_position(const std::vector<float>& obs,int player,int symmetry=0,bool skip=false,double temperature=1,bool randomize=false,double optimism=0) override {
        return delegate_.evaluate_profile(obs,player,symmetry,skip,temperature,randomize,optimism,*profile_);
    }
};
class Backend {
protected:
    GameSpec spec_;
public:
    explicit Backend(GameSpec spec=game_spec(GameKind::GOMOKU)):spec_(spec) {}
    GameSpec spec() const { return spec_; }
    virtual ~Backend() = default;
    virtual bool available() const {return true;}
    virtual bool supports_auxiliary() const { return false; }
    virtual int metadata_channels() const {return 0;}
    virtual ModelDescription description() const {return {};}
    virtual void set_human_profile(const std::string&) {throw std::runtime_error("Model has no human profile");}
    virtual void append_metadata(std::vector<float>&,int) const {}
    virtual void append_profile(std::vector<float>&,int,const SGFMetadata&) const {throw std::runtime_error("Model has no human profile");}
    virtual void validate_human_profile(const std::string&) const {throw std::runtime_error("Model has no human profile");}
    virtual void initialize() {} // Called serially, before any service threads start.
    virtual std::vector<Evaluation> evaluate(const InferenceInputs& inputs) = 0;
};
class InferenceService : public Evaluator {
public:
    std::atomic<uint64_t> requests{0}, batches{0}, max_observed_batch{0};
    std::atomic<uint64_t> cache_hits{0}, submitted{0};
    std::vector<uint64_t> rows_by_server;
    virtual void finish() = 0;
    virtual void drain() = 0;
    virtual void reset_stats() = 0;
    virtual void clear_cache() {}
    virtual void set_human_profile(const std::string&) {throw std::runtime_error("Model has no human profile");}
    virtual ModelDescription description() const {return {};}
};
class BatchEvaluator final : public InferenceService {
    struct Request {
        const BatchEvaluator* owner=nullptr;
        const std::vector<float>* observation=nullptr;
        std::vector<float> transformed;
        std::string cache_key;
        size_t cache_slot = 0;
        int symmetry = 0;
        bool use_cache = false;
        std::mutex mutex;
        std::condition_variable changed;
        bool done=false;
        Evaluation output;
        std::exception_ptr error;
        void complete(Evaluation value) {
            std::lock_guard<std::mutex> lock(mutex);output=std::move(value);done=true;changed.notify_one();
        }
        void fail(std::exception_ptr value) {
            std::lock_guard<std::mutex> lock(mutex);error=value;done=true;changed.notify_one();
        }
    };
    std::vector<std::unique_ptr<Backend>> backends_;
    struct CacheEntry { std::string key; std::shared_ptr<const Evaluation> output; };
    std::vector<CacheEntry> cache_;
    std::deque<std::mutex> cache_locks_;
    std::array<std::vector<int>, 12> mappings_;
    std::string model_;
    bool randomize_;
    int symmetry_;
    std::mt19937_64 random_;
    std::mutex random_mutex_;
    size_t max_batch_, input_size_;
    int wait_us_,canvas_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Request*> queue_;
    size_t active_=0;
    bool closing_ = false;
    std::exception_ptr failure_;
    std::vector<std::thread> servers_;
    void serve(size_t index);
public:
    BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                   size_t max_batch, int wait_us);
    BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends, std::string model, int canvas,
                   size_t max_batch, int wait_us, size_t cache_entries,
                   bool randomize=false, int symmetry=0, uint64_t seed=0,int mutex_pool_power_of_two=16);
    ~BatchEvaluator() override;
    Evaluation evaluate(const std::vector<float>& observation) override;
    Evaluation evaluate_position(const std::vector<float>& observation,int player,int symmetry=0,
                                 bool skip_cache=false,double temperature=1,bool randomize=false,double optimism=0) override;
    Evaluation evaluate_symmetry(const std::vector<float>& observation, int symmetry,
                                 bool skip_cache=false, double policy_temperature=1, bool randomize=false, double policy_optimism=0) override;
    void finish();
    void drain();
    void reset_stats();
    void clear_cache() override;
    void set_human_profile(const std::string&) override;
    bool available() const override {return backends_.front()->available();}
    void validate_human_profile(const std::string& profile) const override {backends_.front()->validate_human_profile(profile);}
    Evaluation evaluate_profile(const std::vector<float>&,int,int,bool,double,bool,double,const SGFMetadata&) override;
    ModelDescription description() const override {return backends_.front()->description();}
};
}
