#pragma once
#include "search/algorithm.h"
#include "search/timecontrols.h"
#include <random>
#include <functional>
#include <limits>
#include <unordered_map>
#include <map>
#include <unordered_set>
namespace etazero {
inline constexpr int64_t SEARCH_CONFIG_LIMIT=int64_t(1)<<50;
struct SearchSettings {
    int64_t simulations;
    int threads;
    double c_puct, virtual_loss, noise_fraction, dirichlet_total_concentration;
    bool reuse_tree;
    int64_t max_visits = 0; // Includes the initial root evaluation and reused visits.
    bool use_fpu = false, shaped_noise = false, policy_target_pruning = false, use_lcb = false;
    bool use_non_buggy_lcb = true;
    double fpu_reduction_max = 0.2, root_fpu_reduction_max = 0, fpu_parent_power = 2;
    double forced_playouts = 0, lcb_stdevs = 5, min_lcb_visit_prop = 0.15;
    double value_weight_exponent = 0.5;
    double chosen_move_subtract = 0, chosen_move_prune = 1;
    double fpu_loss_prop = 0, root_fpu_loss_prop = 0;
    double c_puct_log = 0, c_puct_base = 500;
    double c_puct_stdev_prior = 0.25, c_puct_stdev_prior_weight = 1, c_puct_stdev_scale = 0;
    int root_symmetries = 1;
    double nn_policy_temperature = 1, root_policy_temperature = 1, root_policy_temperature_early = 1;
    double temperature_halflife = 19, chosen_move_temperature_only_below_prob = 1;
    bool nn_randomize = false, fpu_parent_weight_by_visited_policy = true;
    int nn_symmetry = 0;
    int64_t max_playouts = SEARCH_CONFIG_LIMIT;
    double fpu_parent_weight = 0, max_time = 1e20;
    bool graph_search = false;
    int graph_rep_bound = 11;
    double graph_catch_up_leak_prob = 0;
    bool use_uncertainty=false, use_noise_pruning=false;
    double uncertainty_coeff=0.25, uncertainty_exponent=1, uncertainty_max_weight=8;
    double policy_optimism=0, root_policy_optimism=0;
    double noise_prune_utility_scale=0.15, noise_pruning_cap=1e50;
    double subtree_value_bias_factor=0, subtree_value_bias_free_prop=0.8, subtree_value_bias_weight_exponent=0.85;
    int subtree_value_bias_table_num_shards=65536;
    int node_table_shards_power_of_two=16;
    double min_playouts_per_thread=0;
    bool ignore_pre_root_history=false,ignore_all_history=false;
    double wide_root_noise=0,futile_visits_threshold=0;
    int64_t visit_cap_contempt=0;
    int visit_cap_contempt_player=0;
    bool root_symmetry_pruning=false;
    double avoid_repeated_pattern_utility=0;
    std::shared_ptr<const PatternBonusTable> external_patterns;
    std::shared_ptr<Evaluator> human_evaluator;
    bool use_eval_cache=false;int64_t eval_cache_min_visits=100;
    int64_t max_visits_pondering=SEARCH_CONFIG_LIMIT,max_playouts_pondering=SEARCH_CONFIG_LIMIT;
    double max_time_pondering=1e20,lag_buffer=0,tree_reuse_carry_over_time_factor=0;
    double overallocate_time_factor=1,midgame_time_factor=1,midgame_turn_peak_time=130,endgame_turn_time_decay=100;
    double human_sl_cpuct_exploration=1,human_sl_cpuct_permanent=0;
    double human_sl_root_weightless=0,human_sl_root_weightful=0,human_sl_player_weightless=0,human_sl_player_weightful=0,human_sl_opponent_weightless=0,human_sl_opponent_weightful=0;
    double human_sl_chosen_move_proportion=0,human_sl_chosen_move_pikl_lambda=1e9;
    bool human_sl_chosen_move_ignore_pass=false;
    double obvious_moves_time_factor=1,obvious_moves_policy_entropy_tolerance=.30,obvious_moves_policy_surprise_tolerance=.15;
};
void validate_search_controls(const SearchSettings&);
void normalize_selection_weights(std::vector<double>&);
struct EffectiveSearchLimits { int64_t visits, playouts; double seconds; };
EffectiveSearchLimits effective_search_limits(const SearchSettings&, const struct SearchRun&);
int active_search_threads(const SearchSettings&, int64_t playouts);
// Root-only source selection, including virtual losses, pruning, forced visits and noise.
double root_explore_score(double prior,double utility,double weight,int pending,int64_t edge_visits,
    int64_t child_visits,double total,double max_weight,const struct ValueStats& parent,
    const SearchSettings&,bool force,bool hint,double remaining,std::mt19937_64& random);
struct SearchResult;
struct GumbelSettings;
struct SearchRun {
    int64_t max_visits = 0;
    bool training = false, clear_before_search = false, remove_root_noise = false;
    int turn = 0, board_area = 361;
    int64_t max_playouts = -1; // -1 inherits settings; zero issues no new playouts.
    double max_time = -1; // Seconds, including root inference; -1 inherits settings.
    std::function<bool()> should_stop;
    std::array<std::vector<int>,2> avoid_until; // black, white; action-indexed exclusive depth.
    bool avoid_rescale_root=false;
    const TimeControls* time_controls=nullptr;
    bool pondering=false;
    int occupied_points=-1;
    int search_for_player = 0; // 0 uses the root player; pondering sets our fixed player.
    int hint_action = -1; // Root-only guidance; changes always clear statistics.
    bool collect_root_policy_invalid_mass = false;
    // Interactive diagnostics only. Empty callbacks keep training on its existing path.
    bool collect_analysis = false;
    int analysis_pv_length = 32;
    double report_interval = 0.25;
    std::function<void(const SearchResult&)> on_report;
};
// Completed weighted samples: utility and pure W-L use the parent player's perspective.
struct RootChildStats {
    double prior, value_sum, value_sq_sum;
    int64_t visits;
    double weight, weight_sq, draw_sum=0;
    double win_loss_sum=value_sum; // Pure W-L; value_sum/value_sq_sum are search utility.
    double no_result_sum=0;
};
struct ValueStats {
    int64_t visits=0;
    double value=0, value_sq=0, draw=1, weight=0, weight_sq=0;
    double win_loss=value; // Utility may be biased; training/report WDL never uses it.
    double no_result=0;
};
double temperature_at_turn(double early,double late,double halflife,int turn,int board_area);
std::vector<double> policy_temperature_distribution(const std::vector<double>& weights,double temperature);
std::vector<double> temperature_distribution(const std::vector<double>& weights,double temperature,double only_below_prob=1);
double value_weight_cdf(double z);
ValueStats aggregate_values(const WDL& initial,const std::vector<RootChildStats>& children,const SearchSettings& settings,bool noisy_root,double initial_weight=1,SubtreeValueBias* bias=nullptr);
double mixed_policy_logit(double ordinary,double optimistic,double optimism);
std::vector<double> source_policy_probabilities(const Evaluation&,const std::vector<int>& actions,double temperature,double optimism);
double uncertainty_weight(double stdev,bool supported,const SearchSettings& settings);
std::vector<double> noise_pruned_weights(const std::vector<RootChildStats>& children,const SearchSettings& settings);
double explore_scaling(double total_child_weight,const ValueStats& parent,const SearchSettings& settings);
double child_selection_score(double prior,double value,double weight,int pending,double total,
                             const ValueStats& parent,const SearchSettings& settings,bool force);
std::vector<double> noise_alpha_distribution(const std::vector<double>& policy);
double fpu_value(double nn_value, double parent_value, double visited_mass, double power, double reduction,
                 bool by_visited_policy=true, double nn_weight=0);
struct RootSelectionDiagnostic { double weight=0, lcb=0, radius=0; bool has_lcb=false; };
std::vector<double> root_selection_weights(const std::vector<RootChildStats>& children,
                                         const SearchSettings& settings, bool use_lcb, ValueStats parent = {}, bool normalize_output = true,
                                         std::vector<RootSelectionDiagnostic>* diagnostics = nullptr);
std::vector<double> direct_policy_selection(std::vector<double> policy,const SearchSettings&);
struct AnalysisCandidate {
    int action=-1;
    double weight=0, utility=0, win_loss=0;
    WDL wdl{0,1,0};
    bool has_value=false;
    RootSelectionDiagnostic selection;
    int64_t node_visits=0;double node_weight=0;
    std::vector<int> pv;
    std::vector<int64_t> pv_visits,pv_edge_visits;
};
void analysis_policy(SearchResult&,const std::vector<int>&,const std::vector<RootChildStats>&,
                     const SearchSettings&,ValueStats,const GumbelSettings*);
struct SearchResult {
    int action = -1;
    int64_t simulations = 0;
    int64_t root_visits = 0;
    int64_t initial_visits = 0;
    int64_t new_playouts = 0; // Includes a fresh root evaluation, unlike simulations (root edges).
    double seconds = 0;
    bool stopped_early = false;
    double value = 0, utility = 0, policy_surprise = 0;
    double pruned_value=0, pruned_utility=0;
    WDL pruned_wdl{0,1,0};
    double root_policy_invalid_mass = std::numeric_limits<double>::quiet_NaN();
    double network_variance_time=-1;
    double network_sample_weight=0, network_value_stdev=0, search_weight=0, search_weight_sq=0;
    WDL network_wdl{0,1,0}, search_wdl{0,1,0};
    std::vector<double> network_policy, search_policy, human_policy;
    // Selection weights before the final chosen-move temperature.
    std::vector<double> move_policy;
    // Source surprise/entropy use LCB selection with direct HumanSL moves.
    std::vector<double> statistics_policy;
    std::vector<double> policy;
    std::vector<int16_t> policy_target;
    uint64_t graph_hits = 0, graph_catch_ups = 0, graph_cycles = 0;
    size_t graph_nodes = 0;
    std::vector<int64_t> visits;
    // Training-only Q targets: completed child NODE visits (not parent edges)
    // and pure W-L in this root player's perspective.
    std::vector<int64_t> q_visits;
    std::vector<float> q_values;
    std::vector<AnalysisCandidate> analysis;
    bool gumbel=false;
};
// Reported WDL probabilities, with KataGo's floating-point boundary handling.
WDL wdl_from_value(double value, double draw, double no_result=0);
std::vector<int16_t> quantize_policy(const std::vector<double>& weights);
struct SearchEdgeSnapshot {
    int action, perspective, pending;
    int64_t visits;size_t child_id;
};
struct SearchNodeSnapshot {
    size_t id;bool root, ready;int pending;std::string identity;
    ValueStats stats;std::vector<SearchEdgeSnapshot> edges;
};
void mix_secondary_policy(SearchResult&,const std::vector<double>& human,const std::vector<double>& utilities,
    const std::vector<bool>& known,const SearchSettings&,int pass,double target_scale=1);
struct RootFocus {std::vector<int> moves;std::vector<double> weights;double probability=0;};
class SearchTimeBudget;
class GameSearch {
protected:
    std::shared_ptr<const RootFocus> focus_;
    int analysis_pv_length_=32;
    std::shared_ptr<const RootFocus> focus_snapshot() const {return std::atomic_load(&focus_);}
public:
    // Like KataGo, replacement is safe while workers are searching.
    void set_root_focus(std::vector<int> moves,std::vector<double> weights,double probability) {
        if(moves.size()!=weights.size() || !std::isfinite(probability) || probability<0)throw std::runtime_error("Invalid root focus");
        for(size_t i=0;i<moves.size();++i){if(moves[i]<0 || !std::isfinite(weights[i]) || weights[i]<=0)throw std::runtime_error("Invalid root focus move/weight");weights[i]=std::min(weights[i],1e30);}
        std::shared_ptr<const RootFocus> focus;
        if(!moves.empty() && probability>0)focus=std::make_shared<RootFocus>(RootFocus{std::move(moves),std::move(weights),std::min(probability,.99)});
        std::atomic_store(&focus_,std::move(focus));
    }
    virtual ~GameSearch() = default;
    virtual void set_external_patterns(std::shared_ptr<const PatternBonusTable>) {throw std::runtime_error("Pattern history requires AlphaZero");}
    virtual SearchResult run(const Game&, double temperature, SearchRun options = {}) = 0;
    virtual void advance(int action) = 0;
    virtual void reset(uint64_t seed) = 0;
    virtual SearchResult inspect(const Game&,const std::vector<int>&) {throw std::runtime_error("Search inspection unavailable");}
};
}
