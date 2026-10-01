#pragma once
#include "algorithm.h"
#include <random>
namespace etazero {
struct SearchSettings {
    int simulations, threads;
    double c_puct, virtual_loss, noise_fraction, dirichlet_alpha;
    bool reuse_tree;
};
struct SearchResult {
    int action = -1, simulations = 0;
    double value = 0;
    std::vector<double> policy;
    std::vector<int64_t> visits;
};
class Search {
    struct Node;
    struct Edge {
        int action; double prior, search_prior, w = 0; int64_t n = 0; int pending = 0;
        std::unique_ptr<Node> child;
        Edge(int a, double p) : action(a), prior(p), search_prior(p) {}
        ~Edge();
    };
    struct Node {
        std::mutex mutex;
        std::condition_variable changed;
        enum Status { EMPTY, EXPANDING, READY, FAILED } status = EMPTY;
        double initial_value = 0;
        std::exception_ptr error;
        std::vector<std::unique_ptr<Edge>> edges;
    };
    Evaluator* evaluator_ = nullptr;
    SearchSettings settings_;
    std::unique_ptr<Node> root_;
    std::vector<std::mt19937_64> random_;
    std::vector<std::thread> workers_;
    std::mutex work_mutex_, error_mutex_;
    std::condition_variable work_changed_;
    uint64_t generation_ = 0;
    int workers_done_ = 0;
    bool closing_ = false;
    const SearchState* position_ = nullptr;
    std::atomic<int> issued_{0}, completed_{0};
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    bool expand(Node& node, const SearchState& state);
    void simulation(std::mt19937_64& rng);
    void simulate_many(int worker);
    void worker_loop(int worker);
public:
    Search(SearchSettings settings, uint64_t seed);
    Search(Evaluator& evaluator, SearchSettings settings, uint64_t seed);
    ~Search();
    SearchResult run(const Game& game, double temperature);
    SearchResult run(const SearchState& state, double temperature);
    void advance(int action);
    int pending() const;
};
}
