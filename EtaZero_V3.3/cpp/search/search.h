#pragma once
#include "search/search_common.h"
namespace etazero {
class Search : public GameSearch {
protected:
    std::shared_ptr<Evaluator> human_evaluator() const {auto e=settings_.human_evaluator?settings_.human_evaluator:evaluator_?evaluator_->secondary:nullptr;return e&&e->available()?e:nullptr;}
public:
    SearchResult inspect(const Game&,const std::vector<int>&) override;
    void set_external_patterns(std::shared_ptr<const PatternBonusTable> table) override {
        if(settings_.external_patterns!=table)for(auto& shard:eval_cache_)shard.clear();
        settings_.external_patterns=std::move(table);
    }
protected:
    struct Node;
    struct Edge {
        int move_index=-1;
        std::atomic<int64_t> n{0}; std::atomic<int> pending{0}, perspective{0};
        std::atomic<Node*> child{nullptr};
    };
    struct Move {
        int action=-1;
        std::atomic<int> child_index{-1};
    };
    // Published as one immutable snapshot, including the history-specific mask.
    // Go can replace it while other paths still read the previous evaluation.
    struct NodeEvaluation {
        double initial_value=0, initial_weight=1, initial_stdev=0,initial_variance_time=-1;
        bool has_auxiliary=false;
        WDL initial_wdl{0,1,0};
        std::vector<double> human_policy;
        std::vector<double> prior, search_prior; // Negative entries are illegal.
        std::vector<int> prior_order; // Legal, unexpanded move candidates.
    };
    struct CachedEvaluation {
        float value=0,draw=0,weight=0;
        struct First {float value=0,weight=0;};
        std::map<int,First> first_explore;
        double no_result=0;
    };
    struct Node {
        enum Status { EMPTY, EXPANDING, READY, FAILED };
        std::atomic<Status> status{EMPTY};
        size_t id;
        std::string graph_identity,pattern_key;
        std::shared_ptr<const CachedEvaluation> cached_evaluation;
        size_t lock_index;
        explicit Node(size_t index) : id(index), lock_index(index%64) {}
        std::shared_ptr<const NodeEvaluation> evaluation;
        std::shared_ptr<const NodeEvaluation> evaluated() const { return std::atomic_load_explicit(&evaluation,std::memory_order_acquire); }
        void publish(std::shared_ptr<const NodeEvaluation> value) { std::atomic_store_explicit(&evaluation,std::move(value),std::memory_order_release); }
        ValueStats stats;
        int player=1;
        std::vector<double> capped_distribution; // Move-indexed, immutable once installed under stats_mutex.
        SubtreeValueBias subtree_bias;
        std::mutex stats_mutex, update_mutex;
        std::atomic<int> dirty_count{0};
        std::exception_ptr error;
        std::unique_ptr<Move[]> moves;
        size_t move_count=0;
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
    struct TreeDeleter { double free_prop;void operator()(Node* node) const noexcept; };
    struct Sync { std::mutex mutex; std::condition_variable changed; };
    std::array<Sync,64> locks_;
    std::atomic<size_t> next_node_{0};
    struct Visit { Node* parent;Edge* edge;int perspective;bool increment_edge=true;bool weightless=false; };
    struct ThreadState {
        bool counted_playout=true;
        std::unique_ptr<SearchState> state;
        std::vector<Visit> path;
        std::vector<int> candidates;
        std::vector<int> legal_actions;
        std::unordered_set<Node*> graph_path;
        std::unordered_set<size_t> illegal_graph_nodes;
    };
    std::vector<ThreadState> thread_states_;
    Evaluator* evaluator_ = nullptr;
    int root_player_=1,search_for_player_=1;
    std::unique_ptr<PatternBonusTable> patterns_;
    double pattern_bonus(const Node&) const;
    void prepare_patterns(const SearchState&,const SearchRun&);
    void mix_human_policy(SearchResult&,const SearchState&,double temperature,bool training);
    SearchSettings settings_;
    std::unique_ptr<SubtreeValueBiasTable> subtree_bias_table_;
    // Table owns all nodes. Edges and root borrow pointers until quiescent mark-and-sweep.
    struct NodeShard { std::mutex mutex;std::unordered_map<std::string,std::unique_ptr<Node>> nodes; };
    std::deque<NodeShard> node_table_;
    // Published between searches; workers only read immutable entries.
    std::vector<std::unordered_map<std::string,std::shared_ptr<const CachedEvaluation>>> eval_cache_;
    std::string eval_cache_conditions_;
    void prepare_evaluation_cache(const SearchRun&);
    std::shared_ptr<const CachedEvaluation> cached_evaluation(const std::string&) const;
    void record_evaluations();
    void apply_cached_evaluation(Node&,ValueStats&,int64_t visits) const;
    Node* root_=nullptr;
    std::atomic<uint64_t> graph_hits_{0},graph_catch_ups_{0},graph_cycles_{0};
    std::vector<std::mt19937_64> random_;
    std::vector<std::mt19937_64> inference_random_; // Generic states; real NN services own miss-only RNG.
    std::vector<std::thread> workers_;
    std::mutex work_mutex_, error_mutex_;
    std::condition_variable work_changed_;
    uint64_t generation_ = 0;
    int workers_done_ = 0;
    int active_threads_ = 1;
    bool closing_ = false;
    bool cleaning_ = false;
    std::vector<Node*> cleanup_roots_;
    int64_t budget_ = 0;
    bool remove_root_noise_ = false;
    int root_hint_action_ = -1;
    int capped_player_=0;
    std::array<std::vector<int>,2> avoid_until_;
    std::vector<int> root_representative_;
    bool move_allowed(int action,int player,int depth) const;
    bool prepare_restrictions(const SearchState&,const SearchRun&);
    const SearchState* position_ = nullptr;
    int root_symmetries_ = 1;
    std::atomic<int64_t> issued_{0}, completed_{0};
    std::atomic<bool> failed_{false};
    std::atomic<bool> stopped_{false};
    int fresh_playouts_ = 0;
    double max_time_ = 1e20;
    double carried_time_=0,current_temperature_=0;
    std::unique_ptr<SearchTimeBudget> time_budget_;
    void update_time_budget(double elapsed);
    std::chrono::steady_clock::time_point start_;
    std::function<bool()> should_stop_;
    std::function<void()> report_; // Invoked only by coordinator worker 0.
    std::exception_ptr error_;
    enum class Expansion { Ready, Expanded, Contended };
    Expansion expand(Node& node, const SearchState& state, int worker,int depth=0);
    void evaluate_node(Node& node,const SearchState& state,int worker,int symmetries,bool skip_cache=false,int depth=0);
    ValueStats snapshot(Node& node);
    std::vector<RootChildStats> child_stats(Node& node,const NodeEvaluation& eval,bool aggregation=false);
    bool prune_root_children(const SearchState& state);
    void recompute(Node& node,int visits_to_add=0); // Serializes updates without holding parent stats during child reads.
    void update_after_playout(Node& node);
    Node* new_node();
    Node* find_node(const SearchState& state,const SubtreeValueBiasKey* bias_key,const std::string& pattern_key={});
    void prepare_subtree_bias(const SearchState& state);
    bool catch_up(Edge& edge,Node& child,int worker);
    size_t node_count() const;
    Edge& child_edge(Node& node,int move_index);
    void collect_nodes(Node* keep); // Only after all search workers are quiescent.
    template<bool ForcedRoot=false,bool GumbelInterior=false>
    void simulation(int worker,int root_move=-1,const GumbelSettings* gumbel=nullptr);
    int gumbel_selection(Node&,const GumbelSettings&,int depth);
    void simulate_many(int worker);
    void worker_loop(int worker);
    SearchResult analysis_snapshot(int actions,const GumbelSettings* gumbel=nullptr);
public:
    Search(SearchSettings settings, uint64_t seed,bool start_workers=true);
    Search(Evaluator& evaluator, SearchSettings settings, uint64_t seed,bool start_workers=true);
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
