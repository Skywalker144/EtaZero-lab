#pragma once
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
namespace etazero {
// Probabilities are W/D/L from the current player's perspective.
using WDL = std::array<double,3>;
struct Evaluation {
    std::vector<double> logits;
    WDL wdl{0,1,0};
    std::vector<double> optimistic_logits;
    double shortterm_value_stdev=0;
    bool has_auxiliary=false;
    double value() const { return wdl[0]-wdl[2]; }
};
using InferenceInputs = std::vector<const std::vector<float>*>;
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual Evaluation evaluate(const std::vector<float>& observation) = 0;
    // Input and output use canonical board coordinates; symmetry is an NN request option.
    virtual Evaluation evaluate_symmetry(const std::vector<float>& observation, int symmetry,
                                         bool skip_cache=false, double policy_temperature=1, bool randomize=false, double policy_optimism=0);
};
class Backend {
public:
    virtual ~Backend() = default;
    virtual bool supports_auxiliary() const { return false; }
    virtual void initialize() {} // Called on the owning service thread, before taking requests.
    virtual std::vector<Evaluation> evaluate(const InferenceInputs& inputs) = 0;
};
class BatchEvaluator final : public Evaluator {
    struct Request {
        uint64_t id;
        const BatchEvaluator* owner=nullptr;
        const std::vector<float>* observation=nullptr;
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
        std::chrono::steady_clock::time_point submitted;
    };
    std::vector<std::unique_ptr<Backend>> backends_;
    struct CacheEntry { std::string key; std::shared_ptr<const Evaluation> output; };
    std::vector<CacheEntry> cache_;
    std::array<std::mutex, 64> cache_locks_;
    std::string model_;
    bool randomize_;
    int symmetry_;
    std::mt19937_64 random_;
    std::mutex random_mutex_;
    size_t max_batch_, capacity_, input_size_;
    int wait_us_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Request*> queue_;
    size_t active_=0;
    bool closing_ = false;
    std::exception_ptr failure_;
    std::vector<std::thread> servers_;
    void serve(size_t index);
public:
    std::atomic<uint64_t> requests{0}, batches{0}, max_observed_batch{0}, wait_microseconds{0};
    std::atomic<uint64_t> cache_hits{0}, submitted{0};
    std::vector<uint64_t> rows_by_server; // Inspect after finish() joins all owners.
    BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                   size_t max_batch, size_t capacity, int wait_us);
    BatchEvaluator(std::vector<std::unique_ptr<Backend>> backends, std::string model, int canvas,
                   size_t max_batch, size_t capacity, int wait_us, size_t cache_entries,
                   bool randomize=false, int symmetry=0, uint64_t seed=0);
    ~BatchEvaluator() override;
    Evaluation evaluate(const std::vector<float>& observation) override;
    Evaluation evaluate_symmetry(const std::vector<float>& observation, int symmetry,
                                 bool skip_cache=false, double policy_temperature=1, bool randomize=false, double policy_optimism=0) override;
    void finish();
    void drain();
    void reset_stats();
};
}
