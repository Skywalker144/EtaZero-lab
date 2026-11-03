#ifndef TEST_GUMBEL_SYMMETRY_H_
#define TEST_GUMBEL_SYMMETRY_H_

#include "../search/search.h"
#include "../search/searchnode.h"
#include "../core/test.h"

namespace TestGumbelSymmetry {
inline bool representative(const Search& search, Loc loc) {
  return search.rootHistory.rules.isXiangqi() ? !search.rootJointSymDupLoc[loc] :
    loc == Board::PASS_LOC || !search.rootSymDupLoc[loc];
}

inline void check(const Search& search) {
  const auto candidates = search.getGumbelChildren(*search.rootNode);
  const auto* output = search.rootNode->getNNOutput();
  const auto& avoid = search.rootPla == P_BLACK ? search.avoidMoveUntilByLocBlack : search.avoidMoveUntilByLocWhite;
  std::set<Loc> expected, allocated;
  double priorSum = 0;
  int64_t edgeSum = 0;
  for(int p = 0; p < search.policySize; p++) {
    Loc loc = search.getLoc(p);
    const bool allowed = output->getPolicyProbsMaybeNoised()[p] >= 0 &&
      search.rootHistory.isLegal(search.rootBoard,loc,search.rootPla) && representative(search,loc) &&
      (avoid.empty() || avoid[loc] == 0);
    testAssert((candidates[p].prior >= 0) == allowed);
    if(allowed) { expected.insert(loc); priorSum += output->getPolicyProbsMaybeNoised()[p]; }
    testAssert(candidates[p].visits == search.gumbelRoot->visits()[p] && candidates[p].pending == 0);
    edgeSum += candidates[p].visits;
  }
  testAssert(!expected.empty() && search.getRootVisits() == edgeSum+1 && search.gumbelRoot->pending() == 0);
  for(Loc loc : expected) for(int symmetry : search.rootSymmetries) {
    Loc other = search.getSymMove(loc,symmetry);
    if(other != loc) testAssert(expected.count(other) == 0);
  }
  auto children = search.rootNode->getChildren();
  for(int i = 0; i < children.iterateAndCountChildren(); i++) {
    Loc loc = children[i].getMoveLoc();
    testAssert(expected.count(loc) == 1);
    allocated.insert(loc);
  }
  std::vector<Loc> locs; std::vector<double> target;
  testAssert(search.getPolicyTargetValues(*search.rootNode,locs,target));
  testAssert(std::set<Loc>(locs.begin(),locs.end()) == expected);
  testAssert(std::abs(std::accumulate(target.begin(),target.end(),0.0)-1) < 1e-10);
  for(size_t i = 0; i < target.size(); i++) {
    testAssert(std::isfinite(target[i]) && target[i] > 0);
    if(edgeSum == 0) {
      // All completed Qs are the same at zero root action visits. The dense
      // target is the original policy conditioned on source representatives,
      // without summing the priors of discarded symmetric copies.
      const double expectedProb = output->getPolicyProbsMaybeNoised()[search.getPos(locs[i])]/priorSum;
      testAssert(std::abs(target[i]-expectedProb) < 1e-10);
    }
  }
  std::vector<AnalysisData> original, expanded;
  search.getAnalysisData(*search.rootNode,original,0,false,12,false);
  search.getAnalysisData(*search.rootNode,expanded,0,false,12,true);
  for(const auto& data : expanded) {
    const Loc representative = data.isSymmetryOf == Board::NULL_LOC ? data.move : data.isSymmetryOf;
    auto it = std::find_if(original.begin(),original.end(),[&](const AnalysisData& d) { return d.move == representative; });
    testAssert(it != original.end() && allocated.count(representative) == 1);
    testAssert(data.numVisits == it->numVisits && data.pvVisits == it->pvVisits && data.pvEdgeVisits == it->pvEdgeVisits && data.pv.size() == it->pv.size());
    testAssert(data.move == search.getSymMove(representative,data.symmetry));
    for(size_t i = 0; i < data.pv.size(); i++) testAssert(data.pv[i] == search.getSymMove(it->pv[i],data.symmetry));
    testAssert(avoid.empty() || avoid[data.move] == 0);
  }
}

inline void exercise(Search& search, const Board& board, const BoardHistory& history) {
  const SearchParams base = search.searchParams;
  for(int mode : {0,1,2,3,4,5}) {
    if(mode == 5 && !search.humanEvaluator && !search.humanMuZeroEvaluator) continue;
    SearchParams params = base;
    params.rootSymmetryPruning = true;
    params.maxVisits = params.maxPlayouts = mode == 0 ? 1 : 64;
    params.humanSLRootExploreProbWeightless = mode == 5 ? 1 : 0;
    params.humanSLChosenMoveProp = mode == 5 ? 1 : 0;
    search.setParams(params);
    search.setPosition(history.presumedNextMovePla,board,history);
    search.setAvoidMoveUntilByLoc({},{});
    search.setRootSymmetryPruningOnly(mode == 3 ? std::vector<int>{0} : std::vector<int>{});
    search.setRootFocus({},{},0);
    search.beginSearch(false);
    const size_t expectedGroup = mode == 3 ? 1 : history.rules.isXiangqi() || history.rules.game == Rules::Game::HEX ? 2 : board.x_size == board.y_size ? 8 : 4;
    testAssert(search.rootSymmetries.size() == expectedGroup);
    Loc focus = Board::NULL_LOC;
    if(mode == 2 || mode == 4) {
      for(int p = 0; p < search.policySize; p++) {
        Loc loc = search.getLoc(p);
        if(history.isLegal(board,loc,search.rootPla) && representative(search,loc) == (mode == 2)) {
          focus = loc;
          break;
        }
      }
      testAssert(focus != Board::NULL_LOC);
      if(mode == 2) {
        std::vector<int> avoid(search.getMoveLocBufferSize(),0); avoid[focus] = 1;
        search.setAvoidMoveUntilByLoc(search.rootPla == P_BLACK ? avoid : std::vector<int>{},
                                     search.rootPla == P_WHITE ? avoid : std::vector<int>{});
      }
      else search.setRootFocus({focus},{1},.5);
    }
    search.runWholeSearch(search.rootPla);
    check(search);
    testAssert(search.getRootVisits() == (mode == 5 ? 1 : params.maxVisits));
    const Loc chosen = search.getChosenMoveLoc();
    testAssert(history.isLegal(board,chosen,search.rootPla) && representative(search,chosen));
    if(mode == 4) {
      auto children = search.rootNode->getChildren();
      const Loc rep = history.rules.isXiangqi() ? search.rootJointSymRepresentativeLoc[focus] : search.rootSymRepresentativeLoc[focus];
      bool found = false;
      for(int i = 0; i < children.iterateAndCountChildren(); i++) if(children[i].getMoveLoc() == rep) {
        testAssert(children[i].getIfAllocated()->stats.visits.load() > children[i].getEdgeVisits());
        found = true;
      }
      testAssert(found);
    }
    testAssert(search.makeMove(chosen,search.rootPla));
    testAssert(search.rootNode == nullptr && !search.gumbelRoot);
    search.runWholeSearch(search.rootPla);
    check(search);
  }
  search.setParams(base);
}
}
#endif
