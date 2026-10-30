#include "search.h"
#include "searchnode.h"
#include <chrono>

using namespace std;

vector<Gumbel::GumbelChild> Search::getGumbelChildren(const SearchNode& node) const {
  vector<Gumbel::GumbelChild> result(policySize);
  const NNOutput* output = node.getNNOutput();
  const float* priors = output == nullptr ? nullptr : output->getPolicyProbsMaybeNoised();
  const bool isRoot = &node == rootNode;
  const auto& avoid = rootPla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
  double mass = 0;
  for(int pos = 0; pos < policySize; pos++) {
    double p = priors == nullptr ? -1 : priors[pos];
    if(isRoot && p >= 0) {
      Loc loc = getLoc(pos);
      if(!rootHistory.isLegal(rootBoard,loc,rootPla) || !isAllowedRootMove(loc) || (!avoid.empty() && avoid[loc] > 0)) p = -1;
    }
    result[pos].prior = p < 0 ? -1 : std::max(p,std::numeric_limits<double>::min());
    if(p >= 0) mass += result[pos].prior;
  }
  if(mass > 0) for(auto& c : result) if(c.prior >= 0) c.prior /= mass;
  auto children = node.getChildren();
  for(int i = 0; i < children.getCapacity(); i++) {
    const auto& edge = children[i];
    const SearchNode* child = edge.getIfAllocated();
    if(child == nullptr) break;
    auto& entry = result[getPos(edge.getMoveLocRelaxed())];
    if(entry.prior < 0) continue;
    entry.visits = edge.getEdgeVisits();
    entry.q = (node.nextPla == P_WHITE ? 1.0 : -1.0) * child->stats.utilityAvg.load(std::memory_order_acquire);
    entry.pending = child->virtualLosses.load(std::memory_order_acquire);
  }
  return result;
}

bool Search::getPolicyTargetValues(const SearchNode& node, vector<Loc>& locs, vector<double>& values) const {
  if(!searchParams.useGumbel) return getPlaySelectionValues(node,locs,values,nullptr,10.0,false);
  locs.clear(); values.clear();
  const NNOutput* output = node.getNNOutput();
  if(output == nullptr) return false;
  auto children = getGumbelChildren(node);
  bool any = false;
  for(const auto& child : children) any = any || child.prior >= 0;
  if(!any) return false;
  double raw = (node.nextPla == P_WHITE ? 1.0 : -1.0) * getUtilityFromNN(*output);
  auto policy = Gumbel::gumbel_policy(children,raw,searchParams.gumbel);
  for(int pos = 0; pos < policySize; pos++) if(children[pos].prior >= 0) {
    locs.push_back(getLoc(pos));
    values.push_back(policy[pos]);
  }
  return true;
}

Loc Search::getGumbelMove(bool allowPending) const {
  lock_guard<mutex> lock(gumbelMutex);
  if(!gumbelRoot || !rootNode || !rootNode->getNNOutput()) return Board::NULL_LOC;
  auto children = getGumbelChildren(*rootNode);
  double raw = (rootPla == P_WHITE ? 1.0 : -1.0) * getUtilityFromNN(*rootNode->getNNOutput());
  int pos = gumbelRoot->winner(Gumbel::gumbel_completed_q(children,raw,searchParams.gumbel),allowPending);
  return getLoc(pos);
}

Loc Search::getGumbelInteriorMove(const SearchThread& thread, const SearchNode& node) const {
  auto children = getGumbelChildren(node);
  const auto& avoid = thread.pla == P_BLACK ? avoidMoveUntilByLocBlack : avoidMoveUntilByLocWhite;
  if(!avoid.empty()) {
    int depth = muZeroEvaluator ? thread.muZeroDepth : (int)thread.history.moveHistory.size() - (int)rootHistory.moveHistory.size();
    for(int pos = 0; pos < policySize; pos++) if(children[pos].prior >= 0) {
      Loc loc = getLoc(pos);
      if(depth < avoid[loc]) children[pos].prior = -1;
    }
  }
  double raw = (node.nextPla == P_WHITE ? 1.0 : -1.0) * getUtilityFromNN(*node.getNNOutput());
  int pos = Gumbel::gumbel_interior_selection(children,raw,searchParams.gumbel);
  return pos < 0 ? Board::NULL_LOC : getLoc(pos);
}

bool Search::runGumbelPlayout(SearchThread& thread, double upperBoundVisitsLeft) {
  int action = -1;
  bool initial = false;
  {
    unique_lock<mutex> lock(gumbelMutex);
    if(gumbelTerminal) return false;
    if(gumbelInitializing || (gumbelRoot && !gumbelRoot->canReserve())) {
      // Wake on completed work, with a bounded wait so the caller can still observe stop/time controls.
      gumbelReady.wait_for(lock,std::chrono::milliseconds(1));
      return false;
    }
    if(!gumbelRoot) {
      gumbelInitializing = true;
      initial = true;
    }
    else {
      auto children = getGumbelChildren(*rootNode);
      double raw = (rootPla == P_WHITE ? 1.0 : -1.0) * getUtilityFromNN(*rootNode->getNNOutput());
      action = gumbelRoot->reserve(Gumbel::gumbel_completed_q(children,raw,searchParams.gumbel));
      if(action < 0) return false; // Phase barrier or exhausted budget; outer loop checks stop/time limits.
      thread.gumbelRootMove = getLoc(action);
    }
  }
  thread.upperBoundVisitsLeft = upperBoundVisitsLeft;
  thread.shouldCountPlayout = true;
  thread.gumbelRootFocus = false;
  thread.gumbelRootHuman = false;
  auto resetThread = [&]() {
    thread.pla = rootPla; thread.board = rootBoard; thread.history = rootHistory;
    thread.graphHash = rootGraphHash; thread.graphPath.clear();
    thread.muZeroDepth = 0; thread.muZeroParent = nullptr; thread.humanMuZeroParent = nullptr;
    thread.gumbelRootMove = Board::NULL_LOC; thread.gumbelRootFocus = false; thread.gumbelRootHuman = false;
  };
  bool success, counted;
  try {
    if(!initial) {
      Loc focusTarget = sampleRootFocusMove(thread);
      if(focusTarget != Board::NULL_LOC) {
        // Keep one choice through expansion retries. A focus playout is the
        // source weightless extra search: it cancels, rather than consumes,
        // this SH reservation, and never increments the root edge or budget.
        thread.gumbelRootFocus = true;
        thread.shouldCountPlayout = false;
        if(rootNode->getNNOutput()->getPolicyProbsMaybeNoised()[getPos(focusTarget)] >= 0)
          thread.gumbelRootMove = focusTarget;
        // An unselectable target falls back to the reserved action, still
        // weightless and uncounted, just as in the source focus path.
      }
      bool countEdgeVisit = true;
      thread.gumbelRootHuman = sampleHumanSLExplore(thread,*rootNode,true,countEdgeVisit);
    }
    bool propagated = playoutDescend(thread,*rootNode,true);
    counted = thread.shouldCountPlayout;
    success = propagated && counted;
    resetThread();
    lock_guard<mutex> lock(gumbelMutex);
    if(initial) {
      if(success && rootNode->getNNOutput()) {
        auto children = getGumbelChildren(*rootNode);
        vector<double> priors;
        bool any = false;
        for(const auto& child : children) { priors.push_back(child.prior); any = any || child.prior >= 0; }
        if(any) {
          auto settings = searchParams.gumbel;
          if(!searchParams.rootNoiseEnabled) settings.noise_scale = 0;
          gumbelRoot = std::make_unique<Gumbel::GumbelRoot>(priors,gumbelBudget,settings,thread.rand);
        }
        else gumbelTerminal = true;
      }
      else if(success) gumbelTerminal = true;
      gumbelInitializing = false;
    }
    else if(success) gumbelRoot->commit(action);
    else gumbelRoot->cancel(action,true);
    gumbelReady.notify_all();
  }
  catch(...) {
    resetThread();
    lock_guard<mutex> lock(gumbelMutex);
    if(initial) gumbelInitializing = false;
    else gumbelRoot->cancel(action,true);
    gumbelReady.notify_all();
    throw;
  }
  // HumanSL weightless work spends the native playout budget even though it
  // does not propagate to the root. Focus and failed expansions do not.
  return counted;
}
