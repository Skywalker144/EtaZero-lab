#pragma once
#include "algorithm.h"
#include <random>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
namespace etazero {
struct SearchSettings {
    int simulations, threads;
    double c_puct, virtual_loss, noise_fraction, dirichlet_total_concentration;
    bool reuse_tree;
    int max_visits = 0; // Includes the initial root evaluation and reused visits.
    bool use_fpu = false, shaped_noise = false, policy_target_pruning = false, use_lcb = false;
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
    int nn_symmetry = 0, max_playouts = std::numeric_limits<int>::max();
    double fpu_parent_weight = 0, max_time = 1e20;
    bool graph_search = false;
    double graph_catch_up_leak_prob = 0;
    bool use_uncertainty=false, use_noise_pruning=false;
    double uncertainty_coeff=0.25, uncertainty_exponent=1, uncertainty_max_weight=8;
    double policy_optimism=0, root_policy_optimism=0;
    double noise_prune_utility_scale=0.15, noise_pruning_cap=1e50;
};
struct SearchRun {
    int max_visits = 0;
    bool training = false, clear_before_search = false, remove_root_noise = false;
    int turn = 0, board_area = 361;
    int max_playouts = -1; // -1 inherits settings; zero issues no new playouts.
    double max_time = -1; // Seconds, including root inference; -1 inherits settings.
    std::function<bool()> should_stop;
    int hint_action = -1; // Root-only guidance; changes always clear statistics.
};
// Completed weighted samples, all quantities from the parent player's perspective.
struct RootChildStats {
    double prior, value_sum, value_sq_sum;
    int64_t visits;
    double weight, weight_sq, draw_sum=0;
};
struct ValueStats {
    int64_t visits=0;
    double value=0, value_sq=0, draw=1, weight=0, weight_sq=0;
};
double temperature_at_turn(double early,double late,double halflife,int turn,int board_area);
std::vector<double> policy_temperature_distribution(const std::vector<double>& weights,double temperature);
std::vector<double> temperature_distribution(const std::vector<double>& weights,double temperature,double only_below_prob=1);
double value_weight_cdf(double z);
ValueStats aggregate_values(const WDL& initial,const std::vector<RootChildStats>& children,const SearchSettings& settings,bool noisy_root,double initial_weight=1);
double mixed_policy_logit(double ordinary,double optimistic,double optimism);
double uncertainty_weight(double stdev,bool supported,const SearchSettings& settings);
std::vector<double> noise_pruned_weights(const std::vector<RootChildStats>& children,const SearchSettings& settings);
double explore_scaling(double total_child_weight,const ValueStats& parent,const SearchSettings& settings);
double child_selection_score(double prior,double value,double weight,int pending,double total,
                             const ValueStats& parent,const SearchSettings& settings,bool force);
std::vector<double> noise_alpha_distribution(const std::vector<double>& policy);
double fpu_value(double nn_value, double parent_value, double visited_mass, double power, double reduction,
                 bool by_visited_policy=true, double nn_weight=0);
std::vector<double> root_selection_weights(const std::vector<RootChildStats>& children,
                                         const SearchSettings& settings, bool use_lcb, ValueStats parent = {}, bool normalize_output = true);
struct SearchResult {
    int action = -1, simulations = 0;
    int64_t root_visits = 0;
    int64_t initial_visits = 0;
    int new_playouts = 0; // Includes a fresh root evaluation, unlike simulations (root edges).
    double seconds = 0;
    bool stopped_early = false;
    double value = 0, policy_surprise = 0;
    double network_sample_weight=0, network_value_stdev=0, search_weight=0, search_weight_sq=0;
    WDL network_wdl{0,1,0}, search_wdl{0,1,0};
    std::vector<double> network_policy, search_policy;
    std::vector<double> move_policy;
    std::vector<double> policy;
    std::vector<int16_t> policy_target;
    uint64_t graph_hits = 0, graph_catch_ups = 0, graph_cycles = 0;
    size_t graph_nodes = 0;
    std::vector<int64_t> visits;
    // Training-only Q targets: completed child NODE visits (not parent edges)
    // and pure W-L in this root player's perspective.
    std::vector<int64_t> q_visits;
    std::vector<float> q_values;
};
std::vector<int16_t> quantize_policy(const std::vector<double>& weights);
struct SearchEdgeSnapshot {
    int action, perspective, pending;
    int64_t visits;size_t child_id;
};
struct SearchNodeSnapshot {
    size_t id;bool root, ready;int pending;std::string identity;
    ValueStats stats;std::vector<SearchEdgeSnapshot> edges;
};
class Search {
    struct Node;
    struct Edge {
        int move_index=-1;
        std::atomic<int64_t> n{0}; std::atomic<int> pending{0}, perspective{0};
        std::atomic<Node*> child{nullptr};
    };
    struct Move {
        int action=-1;
        std::atomic<int> child_index{-1};
        double prior=0, search_prior=0;
    };
    struct Node {
        enum Status { EMPTY, EXPANDING, READY, FAILED };
        std::atomic<Status> status{EMPTY};
        size_t id;
        std::string graph_identity;
        size_t lock_index;
        explicit Node(size_t index) : id(index), lock_index(index%64) {}
        double initial_value = 0, initial_weight=1, initial_stdev=0;
        bool has_auxiliary=false;
        WDL initial_wdl{0,1,0};
        ValueStats stats;
        std::mutex stats_mutex, update_mutex;
        std::exception_ptr error;
        std::unique_ptr<Move[]> moves;
        size_t move_count=0;
        std::vector<int> prior_order;
        std::array<std::unique_ptr<Edge[]>,3> storage;
        std::array<std::atomic<Edge*>,3> published{};
        std::atomic<int> child_count{0};
        std::atomic<int64_t> visits{0};
        std::atomic<int> pending{0};
        Node* cleanup_next=nullptr;
        Edge& edge(int index) const {
            if(index<8)return published[0].load(std::memory_order_acquire)[index];
            if(index<64)return published[1].load(std::memory_order_acquire)[index-8];
            return published[2].load(std::memory_order_acquire)[index-64];
        }
    };
    struct TreeDeleter { void operator()(Node* node) const noexcept; };
    struct Sync { std::mutex mutex; std::condition_variable changed; };
    std::array<Sync,64> locks_;
    std::atomic<size_t> next_node_{0};
    std::atomic<int> pending_count_{0};
    struct Visit { Node* parent;Edge* edge;Transition transition;bool increment_edge=true; };
    struct ThreadState {
        std::unique_ptr<SearchState> state;
        std::vector<Visit> path;
        std::vector<int> candidates;
        std::unordered_set<Node*> graph_path;
    };
    std::vector<ThreadState> thread_states_;
    Evaluator* evaluator_ = nullptr;
    SearchSettings settings_;
    // Table owns all nodes. Edges and root borrow pointers until quiescent mark-and-sweep.
    struct NodeShard { std::mutex mutex;std::unordered_map<std::string,std::unique_ptr<Node>> nodes; };
    std::array<NodeShard,64> node_table_;
    Node* root_=nullptr;
    std::atomic<uint64_t> graph_hits_{0},graph_catch_ups_{0},graph_cycles_{0};
    std::vector<std::mt19937_64> random_;
    std::vector<std::mt19937_64> inference_random_; // Generic states; real NN services own miss-only RNG.
    std::vector<std::thread> workers_;
    std::mutex work_mutex_, error_mutex_;
    std::condition_variable work_changed_;
    uint64_t generation_ = 0;
    int workers_done_ = 0;
    bool closing_ = false;
    bool cleaning_ = false;
    std::vector<Node*> cleanup_roots_;
    int budget_ = 0;
    bool remove_root_noise_ = false;
    int root_hint_action_ = -1;
    const SearchState* position_ = nullptr;
    int root_symmetries_ = 1;
    std::atomic<int> issued_{0}, completed_{0};
    std::atomic<bool> failed_{false};
    std::atomic<bool> stopped_{false};
    int fresh_playouts_ = 0;
    double max_time_ = 1e20;
    std::chrono::steady_clock::time_point start_;
    std::function<bool()> should_stop_;
    std::exception_ptr error_;
    bool expand(Node& node, const SearchState& state, int worker);
    void evaluate_node(Node& node,const SearchState& state,int worker,int symmetries);
    ValueStats snapshot(Node& node);
    std::vector<RootChildStats> child_stats(Node& node,bool aggregation=false);
    void recompute(Node& node,int visits_to_add=0); // Serializes updates without holding parent stats during child reads.
    Node* new_node();
    Node* find_node(const SearchState& state);
    bool catch_up(Edge& edge,Node& child,int worker);
    size_t node_count() const;
    Edge& child_edge(Node& node,int move_index);
    void collect_nodes(Node* keep); // Only after all search workers are quiescent.
    void simulation(int worker);
    void simulate_many(int worker);
    void worker_loop(int worker);
public:
    Search(SearchSettings settings, uint64_t seed);
    Search(Evaluator& evaluator, SearchSettings settings, uint64_t seed);
    ~Search();
    SearchResult run(const Game& game, double temperature, SearchRun options = {});
    SearchResult run(const SearchState& state, double temperature, SearchRun options = {});
    void advance(int action);
    void reset(uint64_t seed);
    void set_evaluator(Evaluator& evaluator); // Model changes invalidate every tree value/prior.
    int pending() const;
    // Read-only diagnostics; call after run/advance/reset has returned.
    std::vector<SearchNodeSnapshot> inspect_graph();
};
}
