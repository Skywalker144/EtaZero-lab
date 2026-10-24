#pragma once
#include "search/subtree_value_bias_key.h"
#include <map>
#include <memory>
#include <mutex>

namespace etazero {
struct SubtreeValueBiasEntry {
    std::mutex mutex;
    double delta_sum=0, weight_sum=0;
};
// Node-local contribution is replaced on backup, never added once per visit.
// Entries separate player perspectives, so node-perspective W-L is equivalent to
// KataGo's fixed White perspective after converting both delta and utility.
struct SubtreeValueBias {
    std::shared_ptr<SubtreeValueBiasEntry> entry;
    double last_delta=0, last_weight=0;
    double apply(double utility,double factor) const;
    double update(double utility,double child_utility,double original_child_weight,
                  double child_weight,double factor,double exponent);
    void release(double free_prop);
};
class SubtreeValueBiasTable {
    struct Shard {
        std::mutex mutex;
        std::map<SubtreeValueBiasKey,std::shared_ptr<SubtreeValueBiasEntry>> entries;
    };
    size_t count_;
    std::unique_ptr<Shard[]> shards_;
public:
    explicit SubtreeValueBiasTable(int shards);
    std::shared_ptr<SubtreeValueBiasEntry> get(const SubtreeValueBiasKey& key);
    // Only while search and cleanup workers are quiescent, as in KataGo.
    void clear_unused();
    size_t size() const;
    static size_t fixed_bytes(int shards) { return sizeof(Shard)*shards; }
};
}
