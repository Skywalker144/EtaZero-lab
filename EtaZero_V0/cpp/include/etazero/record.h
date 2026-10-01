#pragma once
#include "search.h"
#include <filesystem>
namespace etazero {
struct Step {
    int player, action, simulations;
    float temperature, reward;
    std::vector<float> observation;
    std::vector<double> policy;
    std::vector<int64_t> visits;
};
struct FinishedGame {
    uint64_t id, seed;
    int size, canvas, winner, reason;
    Rule rule;
    std::vector<Step> steps;
    std::vector<float> final_observation;
    int final_player;
};
struct Source { std::string run, attempt, model, config, source; int cycle, worker; };
std::string quote(const std::string& value);
class RecordWriter {
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
    void publish(const std::vector<FinishedGame>& games);
public:
    RecordWriter(std::filesystem::path directory, Source source, size_t rows, size_t capacity, double flush_seconds);
    ~RecordWriter();
    void enqueue(FinishedGame game);
    void finish();
};
}
