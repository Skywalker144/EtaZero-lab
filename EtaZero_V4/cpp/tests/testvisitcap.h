#ifndef TESTS_TESTVISITCAP_H_
#define TESTS_TESTVISITCAP_H_

#include "../core/test.h"
#include "../search/search.h"
#include "../search/searchnode.h"

namespace TestVisitCap {
//Same observable invariant as the pinned source's visit-cap test: after the
//snapshot, uniform-weight visits must remain within one allocation quantum
//of its distribution, with the source allowance for concurrent in-flight work.
inline int checkTree(Search& search) {
  int exercised = 0;
  for(const SearchNode* node : search.enumerateTreePostOrder()) {
    const VisitCapSnapshot* snapshot = node->visitCapSnapshot.load(std::memory_order_acquire);
    int64_t cap = search.getVisitCap(node->nextPla);
    int64_t visits = node->stats.visits.load();
    auto children = node->getChildren();
    int count = children.iterateAndCountChildren();
    testAssert(node->virtualLosses.load() == 0);
    if(cap == 0 || visits < cap || node->getNNOutput() == nullptr || count == 0) {
      testAssert(snapshot == nullptr);
      continue;
    }
    testAssert(snapshot != nullptr);
    if(visits >= 3*cap) exercised++;
    double weight = 0, fractions = 0;
    for(int i = 0; i < count; i++) {
      const auto& edge = children[i];
      weight += edge.getIfAllocated()->stats.getChildWeight(edge.getEdgeVisits());
      fractions += snapshot->getWeightFrac(search.getPos(edge.getMoveLocRelaxed()));
    }
    testAssert(std::abs(fractions-1) < 1e-6);
    double weightPerVisit = weight/std::max<int64_t>(1,visits-1);
    int threads = search.searchParams.numThreads;
    double tolerance = threads == 1 ? weightPerVisit+1e-6 :
      (threads*search.searchParams.numVirtualLossesPerThread+2)*weightPerVisit;
    for(int i = 0; i < count; i++) {
      const auto& edge = children[i];
      double fraction = snapshot->getWeightFrac(search.getPos(edge.getMoveLocRelaxed()));
      double childWeight = edge.getIfAllocated()->stats.getChildWeight(edge.getEdgeVisits());
      if(threads == 1) testAssert(std::abs(fraction*(cap-1)-std::round(fraction*(cap-1))) < 1e-6);
      if(fraction <= 0) testAssert(edge.getEdgeVisits() <= threads);
      testAssert(std::abs(childWeight-fraction*weight) <= tolerance);
    }
  }
  if(search.searchParams.useGumbel) {
    testAssert(search.gumbelRoot && search.gumbelRoot->pending() == 0);
    auto actions = search.getGumbelChildren(*search.rootNode);
    int64_t sum = 0;
    for(size_t i = 0; i < actions.size(); i++) {
      testAssert(actions[i].visits == search.gumbelRoot->visits()[i]);
      testAssert(actions[i].pending == 0);
      sum += actions[i].visits;
    }
    testAssert(search.getRootVisits() == sum+1);
  }
  return exercised;
}
}
#endif
