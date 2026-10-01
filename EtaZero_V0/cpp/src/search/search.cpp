#include "etazero/search.h"
#include <cmath>
#include <functional>
#include <limits>
namespace etazero {
Search::Edge::~Edge() = default;
Search::Search(Evaluator& e, SearchSettings s, uint64_t seed) : Search(s,seed) { evaluator_=&e; }
Search::Search(SearchSettings s, uint64_t seed) : settings_(s), root_(std::make_unique<Node>()) {
    if (s.simulations < 1 || s.threads < 1 || s.c_puct <= 0 || s.virtual_loss < 0 ||
        s.noise_fraction < 0 || s.noise_fraction > 1 || s.dirichlet_alpha <= 0)
        throw std::runtime_error("Invalid search settings");
    for (int i = 0; i < s.threads; ++i) random_.emplace_back(seed + 0x9e3779b97f4a7c15ULL * i);
    try {
        for (int i = 1; i < s.threads; ++i) workers_.emplace_back(&Search::worker_loop, this, i);
    } catch (...) {
        { std::lock_guard<std::mutex> lock(work_mutex_); closing_=true; }
        work_changed_.notify_all();for(auto& worker:workers_)worker.join();throw;
    }
}
Search::~Search() {
    { std::lock_guard<std::mutex> lock(work_mutex_); closing_ = true; }
    work_changed_.notify_all();
    for (auto& t : workers_) t.join();
}
bool Search::expand(Node& node, const SearchState& state) {
    std::unique_lock<std::mutex> lock(node.mutex);
    node.changed.wait(lock, [&] { return node.status != Node::EXPANDING; });
    if (node.status == Node::FAILED) std::rethrow_exception(node.error);
    if (node.status == Node::READY) return false;
    node.status = Node::EXPANDING;
    lock.unlock();
    try {
        auto output = state.evaluate();
        if (output.logits.size() != static_cast<size_t>(state.actions()) || !std::isfinite(output.value))
            throw std::runtime_error("Bad leaf evaluation");
        double maximum = -std::numeric_limits<double>::infinity(), sum = 0;
        for (int a = 0; a < state.actions(); ++a) if (state.legal(a)) maximum = std::max(maximum, output.logits[a]);
        std::vector<std::unique_ptr<Edge>> edges;
        for (int a = 0; a < state.actions(); ++a) if (state.legal(a)) {
            double p = std::exp(output.logits[a] - maximum); sum += p;
            edges.push_back(std::make_unique<Edge>(a, p));
        }
        if (edges.empty() || !std::isfinite(sum) || sum <= 0) throw std::runtime_error("No valid leaf policy");
        for (auto& edge : edges) edge->prior = edge->search_prior = edge->prior / sum;
        lock.lock();
        node.edges = std::move(edges); node.initial_value = output.value; node.status = Node::READY;
        lock.unlock(); node.changed.notify_all();
        return true;
    } catch (...) {
        if (!lock.owns_lock()) lock.lock();
        node.error = std::current_exception(); node.status = Node::FAILED;
        lock.unlock(); node.changed.notify_all(); throw;
    }
}
void Search::simulation(std::mt19937_64& rng) {
    auto state = position_->clone();
    Node* node = root_.get();
    struct Visit { Node* node; Edge* edge; Transition transition; };
    std::vector<Visit> path;
    auto release = [&] {
        for (auto& p : path) { std::lock_guard<std::mutex> lock(p.node->mutex); --p.edge->pending; }
    };
    try {
        double value;
        for (;;) {
            if (state->terminal()) { value = state->terminal_value(); break; }
            if (expand(*node, *state)) { value = node->initial_value; break; }
            std::unique_lock<std::mutex> lock(node->mutex);
            int64_t total = 0;
            for (auto& e : node->edges) total += e->n + e->pending;
            double best = -std::numeric_limits<double>::infinity();
            std::vector<Edge*> candidates;
            for (auto& e : node->edges) {
                double n = e->n + e->pending;
                double q = n ? (e->w - settings_.virtual_loss * e->pending) / n : 0;
                double score = total ? q + settings_.c_puct * e->search_prior * std::sqrt(static_cast<double>(total)) / (1+n)
                                     : e->search_prior;
                if (score > best) { best = score; candidates.clear(); }
                if (score == best) candidates.push_back(e.get());
            }
            Edge* edge = candidates[std::uniform_int_distribution<size_t>(0, candidates.size()-1)(rng)];
            if (!edge->child) edge->child = std::make_unique<Node>();
            path.push_back({node,edge,{}}); ++edge->pending;
            node = edge->child.get(); lock.unlock();
            auto transition=state->move(edge->action);
            if (!std::isfinite(transition.reward) || !std::isfinite(transition.discount) || transition.discount<0 || transition.discount>1 ||
                (transition.perspective!=1 && transition.perspective!=-1))
                throw std::runtime_error("Invalid algorithm transition");
            path.back().transition=transition;
        }
        // Values start at the leaf side-to-move; every edge stores its parent's perspective.
        for (auto p = path.rbegin(); p != path.rend(); ++p) {
            value = p->transition.reward + p->transition.discount * p->transition.perspective * value;
            std::lock_guard<std::mutex> lock(p->node->mutex);
            --p->edge->pending; ++p->edge->n; p->edge->w += value;
        }
        path.clear(); completed_.fetch_add(1);
    } catch (...) { release(); throw; }
}
void Search::simulate_many(int worker) {
    try {
        while (!failed_.load() && issued_.fetch_add(1) < settings_.simulations) simulation(random_[worker]);
    } catch (...) {
        { std::lock_guard<std::mutex> lock(error_mutex_); if (!error_) error_ = std::current_exception(); }
        failed_.store(true);
    }
}
void Search::worker_loop(int worker) {
    uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lock(work_mutex_);
        work_changed_.wait(lock, [&] { return closing_ || generation_ != seen; });
        if (closing_) return;
        seen = generation_; lock.unlock(); simulate_many(worker); lock.lock();
        ++workers_done_; lock.unlock(); work_changed_.notify_all();
    }
}
SearchResult Search::run(const Game& game, double temperature) {
    if (!evaluator_) throw std::runtime_error("Game search requires an AlphaZero evaluator");
    AlphaZeroState state(game,*evaluator_);
    return run(state,temperature);
}
SearchResult Search::run(const SearchState& state, double temperature) {
    if (state.terminal() || temperature < 0 || !std::isfinite(temperature)) throw std::runtime_error("Invalid search position/temperature");
    expand(*root_, state); // root expansion does not consume a simulation
    {
        std::lock_guard<std::mutex> lock(root_->mutex);
        std::gamma_distribution<double> gamma(settings_.dirichlet_alpha, 1);
        std::vector<double> noise; double sum = 0;
        for (auto& e : root_->edges) { (void)e; noise.push_back(settings_.noise_fraction ? gamma(random_[0]) : 1); sum += noise.back(); }
        if (sum <= 0 || !std::isfinite(sum)) throw std::runtime_error("Invalid Dirichlet draw");
        for (size_t i = 0; i < noise.size(); ++i) {
            auto& e = root_->edges[i];
            e->search_prior = (1-settings_.noise_fraction)*e->prior + settings_.noise_fraction*noise[i]/sum;
        }
    }
    { std::lock_guard<std::mutex> lock(work_mutex_);
      position_ = &state; issued_ = 0; completed_ = 0; failed_ = false; error_ = nullptr;
      workers_done_ = 0; ++generation_; }
    work_changed_.notify_all(); simulate_many(0);
    { std::unique_lock<std::mutex> lock(work_mutex_);
      work_changed_.wait(lock, [&] { return workers_done_ == static_cast<int>(workers_.size()); }); }
    if (pending() != 0) throw std::runtime_error("Search leaked pending visits");
    if (error_) std::rethrow_exception(error_);
    if (completed_ != settings_.simulations) throw std::runtime_error("Search budget mismatch");
    SearchResult result; result.simulations = completed_;
    result.policy.assign(state.actions(), 0); result.visits.assign(state.actions(), 0);
    int64_t total = 0, max_n = 0; double w = 0;
    for (auto& e : root_->edges) { total += e->n; w += e->w; max_n = std::max(max_n, e->n); }
    if (total < 1) throw std::runtime_error("Search has no completed root visits");
    std::vector<double> behavior;
    for (auto& e : root_->edges) {
        result.visits[e->action] = e->n; result.policy[e->action] = static_cast<double>(e->n) / total;
        behavior.push_back(temperature == 0 ? (e->n == max_n ? 1.0 : 0.0)
                            : (e->n ? std::exp((std::log(static_cast<double>(e->n))-std::log(static_cast<double>(max_n)))/temperature) : 0.0));
    }
    result.action = root_->edges[std::discrete_distribution<size_t>(behavior.begin(), behavior.end())(random_[0])]->action;
    result.value = w / total;
    return result;
}
void Search::advance(int action) {
    std::unique_ptr<Node> next;
    if (settings_.reuse_tree) for (auto& e : root_->edges) if (e->action == action) next = std::move(e->child);
    root_ = next ? std::move(next) : std::make_unique<Node>();
    // Root-only exploration must not follow a promoted node into the nonroot tree.
    for (auto& e : root_->edges) e->search_prior = e->prior;
}
int Search::pending() const {
    std::function<int(const Node*)> count = [&](const Node* n) {
        int result = 0;
        for (auto& e : n->edges) { result += e->pending; if (e->child) result += count(e->child.get()); }
        return result;
    };
    return count(root_.get());
}
}
