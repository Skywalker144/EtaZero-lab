#pragma once
#include "search/search_common.h"
#include <optional>
namespace etazero {
// Search-thread 0 refreshes at most ten times/second; other workers only read these limits.
class SearchTimeBudget {
    SearchSettings settings_;std::optional<TimeControls> clock_;
    int area_,turn_,occupied_;bool pondering_;
    double carried_,cap_,last_update_=-1;
    std::atomic<double> limit_,visits_left_{1e30};
public:
    SearchTimeBudget(const SearchSettings&,const SearchRun&,double carried);
    void update(double elapsed,int64_t root_visits,const std::vector<RootChildStats>&,
                const std::vector<double>& root_policy,ValueStats parent,double temperature,const SearchResult* selection=nullptr);
    bool needs_update(double elapsed) const {return !pondering_ && clock_ && (last_update_<0 || elapsed>=last_update_+.1);}
    double limit() const {return limit_.load(std::memory_order_acquire);}
    double visits_left() const {return visits_left_.load(std::memory_order_acquire);}
    static double estimate_visits_left(int64_t visits,double elapsed,double limit,double carried,int threads);
};
}
