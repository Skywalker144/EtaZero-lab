#include "etazero/inference.h"
#include "etazero/schema.h"
#include <cmath>
#include <stdexcept>
namespace etazero {
BatchEvaluator::BatchEvaluator(std::unique_ptr<Backend> backend, std::string model, int canvas,
                               size_t max_batch, size_t capacity, int wait_us)
    : backend_(std::move(backend)), model_(std::move(model)), max_batch_(max_batch),
      capacity_(capacity), input_size_(INPUT_PLANES * canvas * canvas), wait_us_(wait_us) {
    if (!backend_ || model_.empty() || !max_batch || !capacity || wait_us < 0)
        throw std::runtime_error("Invalid inference service configuration");
    server_ = std::thread(&BatchEvaluator::serve, this);
}
BatchEvaluator::~BatchEvaluator() { try { finish(); } catch (...) {} }
void BatchEvaluator::finish() {
    { std::lock_guard<std::mutex> lock(mutex_); closing_ = true; }
    changed_.notify_all();
    if (server_.joinable()) server_.join();
    if (failure_) std::rethrow_exception(failure_);
}
Evaluation BatchEvaluator::evaluate(const std::vector<float>& obs) {
    if (obs.size() != input_size_) throw std::runtime_error("Inference input shape mismatch");
    auto request = std::make_shared<Request>();
    static std::atomic<uint64_t> next{0};
    request->id = next.fetch_add(1); request->model = model_; request->observation = obs;
    request->submitted = std::chrono::steady_clock::now();
    auto future = request->result.get_future();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return queue_.size() < capacity_ || closing_ || failure_; });
        if (failure_) std::rethrow_exception(failure_);
        if (closing_) throw std::runtime_error("Inference service is closing");
        queue_.push_back(request);
    }
    changed_.notify_all();
    return future.get();
}
void BatchEvaluator::serve() {
    for (;;) {
        std::vector<std::shared_ptr<Request>> batch;
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
        }
        changed_.notify_all();
        try {
            std::vector<std::vector<float>> inputs;
            auto now = std::chrono::steady_clock::now();
            for (const auto& r : batch) {
                if (r->model != model_ || r->observation.size() != input_size_)
                    throw std::runtime_error("Mixed inference contract/model in batch");
                wait_microseconds.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(now-r->submitted).count());
                inputs.push_back(std::move(r->observation));
            }
            auto outputs = backend_->evaluate(inputs);
            if (outputs.size() != batch.size()) throw std::runtime_error("Inference output batch mismatch");
            // Validate the whole batch before fulfilling any promises, so failure wakes everyone once.
            for (const auto& out : outputs) {
                if (out.logits.size() * INPUT_PLANES != input_size_ || !std::isfinite(out.value) || std::abs(out.value) > 1.00001)
                    throw std::runtime_error("Invalid inference output value/shape");
                for (double p : out.logits) if (!std::isfinite(p)) throw std::runtime_error("Nonfinite policy logits");
            }
            requests.fetch_add(batch.size()); batches.fetch_add(1);
            max_observed_batch.store(std::max<uint64_t>(max_observed_batch.load(), batch.size()));
            for (size_t i = 0; i < batch.size(); ++i) batch[i]->result.set_value(std::move(outputs[i]));
        } catch (...) {
            auto error = std::current_exception();
            { std::lock_guard<std::mutex> lock(mutex_);
              failure_ = error;
              for (auto& r : batch) r->result.set_exception(error);
              for (auto& r : queue_) r->result.set_exception(error);
              queue_.clear(); }
            changed_.notify_all();
            return;
        }
    }
}
}
