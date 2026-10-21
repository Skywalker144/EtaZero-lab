#include "etazero/subtree_value_bias.h"
#include "etazero/game.h"
#include <cmath>

namespace etazero {
SubtreeValueBiasKey subtree_value_bias_key(const Game& before,int action) {
    if(before.last_action()<0 || !before.legal(action))throw std::runtime_error("Subtree value bias needs two real legal moves");
    int x=action%before.canvas(),y=action/before.canvas();
    SubtreeValueBiasKey key;
    for(int dy=-2;dy<=2;++dy)for(int dx=-2;dx<=2;++dx) {
        int xx=x+dx,yy=y+dy,color=3;
        if(xx>=0 && yy>=0 && xx<before.size() && yy<before.size()) {
            int cell=before.board().cells[yy*before.size()+xx];color=cell==0?0:cell==1?1:2;
        }
        key.pattern=(key.pattern<<2)|static_cast<uint64_t>(color);
    }
    // Use board coordinates, independent of the NN canvas (KataGo Location).
    auto location=[&](int a){return (a/before.canvas()+1)*(before.size()+1)+a%before.canvas()+1;};
    key.moves=static_cast<uint64_t>(location(before.last_action())) |
              (static_cast<uint64_t>(location(action))<<16) |
              (static_cast<uint64_t>(before.player()==1?2:1)<<32);
    return key;
}
double SubtreeValueBias::apply(double utility,double factor) const {
    if(!entry || factor==0)return utility;
    std::lock_guard<std::mutex> lock(entry->mutex);
    return entry->weight_sum>0.001?utility+factor*entry->delta_sum/entry->weight_sum:utility;
}
double SubtreeValueBias::update(double utility,double child_utility,double original_child_weight,
                               double child_weight,double factor,double exponent) {
    if(!entry || factor==0)return utility;
    std::lock_guard<std::mutex> lock(entry->mutex);
    if(child_weight>1e-10) {
        double weight=std::pow(original_child_weight,exponent);
        double delta=(child_utility-utility)*weight;
        entry->delta_sum+=delta-last_delta;entry->weight_sum+=weight-last_weight;
        last_delta=delta;last_weight=weight;
    }
    return entry->weight_sum>0.001?utility+factor*entry->delta_sum/entry->weight_sum:utility;
}
void SubtreeValueBias::release(double free_prop) {
    if(!entry)return;
    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        entry->delta_sum-=last_delta*free_prop;entry->weight_sum-=last_weight*free_prop;
    }
    entry.reset();last_delta=last_weight=0;
}
SubtreeValueBiasTable::SubtreeValueBiasTable(int shards) : count_(shards) {
    if(shards<1)throw std::runtime_error("Subtree value bias needs positive table shards");
    shards_=std::make_unique<Shard[]>(count_);
}
std::shared_ptr<SubtreeValueBiasEntry> SubtreeValueBiasTable::get(const SubtreeValueBiasKey& key) {
    // Sharding changes lock contention only; it never limits or aliases keys.
    uint64_t hash=key.pattern^(key.moves+0x9e3779b97f4a7c15ULL+(key.pattern<<6)+(key.pattern>>2));
    auto& shard=shards_[hash%count_];std::lock_guard<std::mutex> lock(shard.mutex);
    auto& slot=shard.entries[key];if(!slot)slot=std::make_shared<SubtreeValueBiasEntry>();return slot;
}
void SubtreeValueBiasTable::clear_unused() {
    for(size_t i=0;i<count_;++i) {
        auto& entries=shards_[i].entries;
        for(auto it=entries.begin();it!=entries.end();) {
            if(it->second.use_count()==1)it=entries.erase(it);else ++it;
        }
    }
}
size_t SubtreeValueBiasTable::size() const {
    size_t size=0;for(size_t i=0;i<count_;++i)size+=shards_[i].entries.size();return size;
}
}
