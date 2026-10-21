#pragma once
#include "search.h"

namespace etazero {
// Mctx 428bfb7: completed-Q + mixed value, node-local min/max rescaling.
struct GumbelSettings {
    int max_num_considered_actions=16;
    double c_visit=50, c_scale=0.1, noise_scale=1;
    bool rescale_q_values=true;
    bool sample_visits=false; // Explicit paper-style explorative behavior; targets remain pi'.
    void validate() const;
};
struct GumbelChild {
    double prior=0, q=0;
    int64_t visits=0;
    int pending=0;
};
std::vector<double> gumbel_completed_q(const std::vector<GumbelChild>&,double raw_value,const GumbelSettings&);
std::vector<double> gumbel_policy(const std::vector<GumbelChild>&,double raw_value,const GumbelSettings&);
int gumbel_interior_selection(const std::vector<GumbelChild>&,double raw_value,const GumbelSettings&);
std::vector<int16_t> quantize_gumbel_policy(const std::vector<double>&);

// One root search owns this scheduler. Its caller serializes reserve/commit.
// Completed counts define Q; reserved counts prevent duplicate/over-budget work.
class GumbelRoot {
    struct Phase { int64_t begin, length; int actions, first_visit; };
    std::vector<Phase> phases_;
    std::vector<int64_t> visits_;
    std::vector<int> pending_;
    std::vector<double> logits_, noise_;
    int budget_=0, issued_=0, pending_total_=0, considered_=-1;
public:
    GumbelRoot(const std::vector<double>& priors,int budget,const GumbelSettings&,std::mt19937_64&);
    int considered_visit(int simulation) const;
    bool done() const { return issued_>=budget_; }
    int reserve(const std::vector<double>& transformed_q);
    void commit(int action);
    void cancel(int action);
    int winner(const std::vector<double>& transformed_q) const;
    const std::vector<int64_t>& visits() const { return visits_; }
    int pending() const { return pending_total_; }
};
void finish_gumbel_result(SearchResult&,const std::vector<int>& actions,
                          const std::vector<GumbelChild>&,double raw_value,
                          const GumbelRoot&,const GumbelSettings&,double temperature,
                          std::mt19937_64&);
}
