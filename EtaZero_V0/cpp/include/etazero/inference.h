#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
namespace etazero {
struct Evaluation { std::vector<double> logits; double value = 0; };
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual Evaluation evaluate(const std::vector<float>& observation) = 0;
};
class Backend {
public:
    virtual ~Backend() = default;
    virtual std::vector<Evaluation> evaluate(const std::vector<std::vector<float>>& inputs) = 0;
};
class BatchEvaluator final : public Evaluator {
    struct Request {
        uint64_t id;
        std::string model;
        std::vector<float> observation;
        std::promise<Evaluation> result;
        std::chrono::steady_clock::time_point submitted;
    };
    std::unique_ptr<Backend> backend_;
    std::string model_;
    size_t max_batch_, capacity_, input_size_;
    int wait_us_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::shared_ptr<Request>> queue_;
    bool closing_ = false;
    std::exception_ptr failure_;
    std::thread server_;
    void serve();
public:
    std::atomic<uint64_t> requests{0}, batches{0}, max_observed_batch{0}, wait_microseconds{0};
    BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                   size_t max_batch, size_t capacity, int wait_us);
    ~BatchEvaluator() override;
    Evaluation evaluate(const std::vector<float>& observation) override;
    void finish();
};
}
