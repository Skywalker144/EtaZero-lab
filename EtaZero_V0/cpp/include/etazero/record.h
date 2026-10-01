#pragma once
#include "opening.h"
#include <filesystem>
namespace etazero {
struct Step {
    int player, action, simulations;
    float temperature, reward;
    std::vector<float> observation;
    std::vector<double> policy;
    std::vector<int64_t> visits;
    bool trainable = true;
    bool cheap_search = false;
    double target_weight = 1, policy_surprise = 0, value_surprise = 0;
    int row_repeats = 1;
    WDL network_wdl{0,1,0}, search_wdl{0,1,0};
};
struct FinishedGame {
    uint64_t id, seed;
    int size, canvas, winner, reason;
    Rule rule;
    std::vector<Step> steps;
    std::vector<float> final_observation;
    int final_player;
    OpeningResult opening;
};
void apply_training_weights(FinishedGame& game, double policy_factor, double value_factor, std::mt19937_64& rng);
struct Source { std::string run, attempt, model, config, source; int iteration, worker; };
std::string quote(const std::string& value);
class RecordWriter {
    struct Buffers;
    std::unique_ptr<Buffers> buffers_;
    std::filesystem::path directory_;
    Source source_;
    size_t max_rows_, capacity_;
    double flush_seconds_;
    unsigned next_shard_ = 0;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<FinishedGame> queue_;
    bool closing_ = false;
    std::exception_ptr failure_;
    std::thread worker_;
    void loop();
    void append(const FinishedGame& game);
    void publish();
public:
    RecordWriter(std::filesystem::path directory, Source source, size_t rows, size_t capacity, double flush_seconds);
    ~RecordWriter();
    void enqueue(FinishedGame game);
    void finish();
};
}
