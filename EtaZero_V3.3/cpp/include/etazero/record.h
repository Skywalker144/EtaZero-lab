#pragma once
#include "opening.h"
#include <filesystem>
namespace etazero {
struct SearchMetadata {
    double policy_entropy=0,search_entropy=0,raw_policy_entropy=0,raw_winloss=0,policy_surprise=0;
    int64_t visits=0;
    int turn=0;
    std::array<float,5> rules{};
};
SearchMetadata search_metadata(const SearchResult&,const Game&);
struct Step {
    int player, action;
    int64_t simulations;
    float temperature, reward;
    std::vector<float> observation;
    std::vector<double> policy;
    std::vector<int64_t> visits;
    bool trainable = true;
    bool cheap_search = false;
    double target_weight = 1, policy_surprise = 0, value_surprise = 0;
    int row_repeats = 1;
    WDL network_wdl{0,1,0}, search_wdl{0,1,0};
    std::vector<int16_t> policy_target;
    bool reanalyzed=false, reanalysis_used_outcome=true;
    int64_t reanalysis_original_visits=0;
    double reanalysis_policy_surprise=0,reanalysis_value_surprise=0;
    std::vector<float> q_values;
    std::vector<int64_t> q_visits;
    SearchMetadata metadata;
};
struct SidePosition {
    int player;
    std::vector<float> observation;
    std::vector<int16_t> policy_target;
    std::vector<int64_t> visits;
    WDL search_wdl;
    double target_weight=1;
    int row_repeats=1;
    std::vector<float> q_values;
    std::vector<int64_t> q_visits;
    SearchMetadata metadata;
};
struct FinishedGame {
    uint64_t id, seed;
    int size, canvas, winner=0;
    Reason reason=Reason::DRAW;
    Outcome outcome() const { return {winner,reason}; }
    Rule rule;
    RuleOptions options;
    std::vector<Step> steps;
    std::vector<float> final_observation;
    int final_player;
    OpeningResult opening;
    double forbidden_feature_dropout_prob = 0.5;
    double training_weight = 1.0;
    std::vector<SidePosition> side_positions;
    double root_policy_invalid_mass_sum = 0;
    uint64_t root_policy_invalid_mass_count = 0;
};
void compute_value_surprises(FinishedGame& game,bool direct=false);
void apply_training_weights(FinishedGame& game,double policy_factor,double value_factor,std::mt19937_64& rng,
                            bool direct=false,bool use_reanalyze=false,double scale_data_weight=1.0);
// Source trainingwrite.cpp's float32 stochastic rounding, without Go score Q.
int16_t quantize_q_value(float winloss,std::mt19937_64& rng);
size_t first_file_row_limit(size_t maximum,double minimum_proportion,double uniform);
struct Source { std::string run, attempt, model, config, source; int iteration, worker; int unroll_steps=0; bool gumbel=false,full_gumbel=false,research_records=false; };
std::string quote(const std::string& value);
class RecordWriter {
    struct Buffers;
    std::unique_ptr<Buffers> buffers_;
    std::filesystem::path directory_;
    Source source_;
    size_t max_rows_, first_rows_, capacity_;
    bool first_file_=true;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<FinishedGame> queue_;
    bool closing_ = false;
    std::exception_ptr failure_;
    std::thread worker_;
    void loop();
    void append(const FinishedGame& game, size_t row_begin, size_t rows);
    void publish();
public:
    RecordWriter(std::filesystem::path directory, Source source, size_t rows, size_t capacity,
                 double first_file_min_random_proportion=0.15, uint64_t seed=0);
    ~RecordWriter();
    void enqueue(FinishedGame game);
    void finish();
};
}
