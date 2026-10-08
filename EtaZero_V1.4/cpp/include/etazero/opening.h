#pragma once

#include "search.h"
#include "config.h"
#include <functional>

namespace etazero {

struct OpeningConfig {
    double probability, avg_dist_factor, balance_exponent, rejection_probability, rejection_probability_fallback;
    int max_tries;
    double hex_probability, hex_make_fair_probability, hex_balance_exponent, hex_min_accept_rate;
    bool policy_init, policy_after, policy_on_failure;
    double policy_init_mean, policy_temperature;
    explicit OpeningConfig(const Config& c, const std::string& policy_section = "opening");
};

enum class OpeningStatus { NotAttempted, Success, Failed, Interrupted };
double hex_opening_accept_rate(double white_win_probability, double exponent, double minimum);

struct OpeningResult {
    OpeningStatus status = OpeningStatus::NotAttempted;
    int attempts = 0, balanced_moves = 0, policy_moves = 0;
    double start_value = 0;
    std::string failure;
    std::vector<int> actions;
    // 0=reference Black bot, 1=reference White bot; same bot throughout each balance attempt.
    std::vector<int> balance_evaluators, policy_evaluators;
    int initial_position_moves=0, initial_position_kind=0, hint_action=-1;
};

OpeningResult initialize_opening(Game& game, const OpeningConfig& config, Evaluator& black, Evaluator& white,
                                 std::mt19937_64& random, const std::function<bool()>& cancelled = {});

OpeningResult initialize_opening(Game& game, const OpeningConfig& config, Evaluator& evaluator,
                                 std::mt19937_64& random, const std::function<bool()>& cancelled = {});

}
