#ifndef SEARCH_EVALCACHE_H_
#define SEARCH_EVALCACHE_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/board.h"
#include "../search/mutexpool.h"

struct FirstExploreEval {
  float avgWinLoss;
  float cacheWeight;
  FirstExploreEval();
  FirstExploreEval(float avgWinLoss, float cacheWeight);
};

struct SearchNode;

struct EvalCacheEntry {
  float avgWinLoss;
  float avgDraw;
  float avgNoResult;
  float cacheWeight;
  std::map<Loc,FirstExploreEval> firstExploreEvals;

  EvalCacheEntry();
};

struct EvalCacheTable {
  std::vector<std::map<Hash128,std::shared_ptr<EvalCacheEntry>>> entries;
  MutexPool* mutexPool;

  EvalCacheTable(int32_t numShards);
  ~EvalCacheTable();

  // These are all thread-safe, and EvalCacheEntry is only ever updated via replacement, not mutation
  // similar to nnevals.
  std::shared_ptr<EvalCacheEntry> find(Hash128 graphHash);
  void update(Hash128 graphHash, const SearchNode* node, int64_t evalCacheMinVisits);
  void clear();
};



#endif // SEARCH_EVALCACHE_H_
