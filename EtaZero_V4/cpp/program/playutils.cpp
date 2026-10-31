#include "../program/playutils.h"
#include "../program/selfplaymanager.h"

#include <sstream>

#include "../core/timer.h"
#include "../core/test.h"
#include "../game/xiangqi.h"
#include "../search/policybuffer.h"

using namespace std;

static int getDefaultMaxExtraBlack(double sqrtBoardArea) {
  if(sqrtBoardArea <= 10.00001)
    return 0;
  if(sqrtBoardArea <= 14.00001)
    return 1;
  if(sqrtBoardArea <= 16.00001)
    return 2;
  if(sqrtBoardArea <= 17.00001)
    return 3;
  if(sqrtBoardArea <= 18.00001)
    return 4;
  return 5;
}

int PlayUtils::chooseExtraBlack(double handicapProb, int numExtraBlackFixed, double sqrtBoardArea, Rand& rand) {
  int defaultMaxExtraBlack = getDefaultMaxExtraBlack(sqrtBoardArea);
  if((numExtraBlackFixed > 0 || defaultMaxExtraBlack > 0) && rand.nextBool(handicapProb))
    return numExtraBlackFixed > 0 ? numExtraBlackFixed : 1 + rand.nextUInt(defaultMaxExtraBlack);
  return 0;
}

Loc PlayUtils::chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove) {
  int numLegalMoves = 0;
  SearchPolicyBuffer<Loc,Board::MAX_ARR_SIZE> storage(hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS : Board::MAX_ARR_SIZE);
  Loc* locs = storage.data();
  testAssert(pla == hist.presumedNextMovePla);
  if(hist.rules.isXiangqi() && hist.isGameFinished) return Board::NULL_LOC;
  if(hist.rules.isXiangqi()) {
    for(int action : hist.getXiangqiState().legalActions()) {
      Loc loc = Xiangqi::actionToLoc(action);
      if(loc != banMove) locs[numLegalMoves++] = loc;
    }
  }
  else for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    if(hist.isLegal(board,loc,pla) && loc != banMove) {
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }
  if(numLegalMoves > 0) {
    int n = gameRand.nextUInt(numLegalMoves);
    return locs[n];
  }
  return Board::NULL_LOC;
}

int PlayUtils::chooseRandomLegalMoves(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc* buf, int len) {
  int numLegalMoves = 0;
  SearchPolicyBuffer<Loc,Board::MAX_ARR_SIZE> storage(hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS : Board::MAX_ARR_SIZE);
  Loc* locs = storage.data();
  testAssert(pla == hist.presumedNextMovePla);
  if(hist.rules.isXiangqi() && hist.isGameFinished) return 0;
  if(hist.rules.isXiangqi()) {
    for(int action : hist.getXiangqiState().legalActions())
      locs[numLegalMoves++] = Xiangqi::actionToLoc(action);
  }
  else for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    if(hist.isLegal(board,loc,pla)) {
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }
  if(numLegalMoves > 0) {
    for(int i = 0; i<len; i++) {
      int n = gameRand.nextUInt(numLegalMoves);
      buf[i] = locs[n];
    }
    return len;
  }
  return 0;
}


Loc PlayUtils::chooseRandomPolicyMove(
  const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, double temperature, bool allowPass, Loc banMove
) {
  const float* policyProbs = nnOutput->policyProbs;
  int nnXLen = nnOutput->nnXLen;
  int nnYLen = nnOutput->nnYLen;
  int numLegalMoves = 0;
  int policySize = nnOutput->getPolicyStorageSize();
  SearchPolicyBuffer<double> relProbStorage(policySize);
  SearchPolicyBuffer<Loc> locStorage(policySize);
  double* relProbs = relProbStorage.data();
  Loc* locs = locStorage.data();
  testAssert(pla == hist.presumedNextMovePla);
  for(int pos = 0; pos<policySize; pos++) {
    Loc loc = hist.rules.isXiangqi() ? Xiangqi::actionToLoc(pos) :
      NNPos::posToLoc(pos,board.x_size,board.y_size,nnXLen,nnYLen);
    if((loc == Board::PASS_LOC && !allowPass) || loc == banMove)
      continue;
    if(policyProbs[pos] > 0.0 && hist.isLegal(board,loc,pla)) {
      double relProb = policyProbs[pos];
      relProbs[numLegalMoves] = relProb;
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }

  //Just in case the policy map is somehow not consistent with the board position
  if(numLegalMoves > 0) {
    double onlyBelowProb = 1.0;
    uint32_t n = Search::chooseIndexWithTemperature(gameRand, relProbs, numLegalMoves, temperature, onlyBelowProb, NULL);
    return locs[n];
  }
  return Board::NULL_LOC;
}


Loc PlayUtils::getGameInitializationMove(
  Search* botB, Search* botW, const Board& board, const BoardHistory& hist, Player pla, NNResultBuf& buf,
  Rand& gameRand, double temperature
) {
  const Search* searcher = (pla == P_BLACK ? botB : botW);
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = searcher->searchParams.drawEquivalentWinsForWhite;
  //Featurize for this bot's net the way that bot's own searches would, even if the game-level history
  //carries different BoardHistoryModes.
  nnInputParams.passAliveSuicideRulesOverride =
    searcher->mainEvaluatorHistoryModes().alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
  nnInputParams.excludeTerritoryAdjAtariOverride =
    searcher->mainEvaluatorHistoryModes().excludeTerritoryAdjacentToAtari ? 1 : 0;
  searcher->evaluatePosition(board,hist,pla,nnInputParams,buf,false);
  std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);

  vector<Loc> locs;
  vector<double> playSelectionValues;
  int nnXLen = nnOutput->nnXLen;
  int nnYLen = nnOutput->nnYLen;
  testAssert(nnXLen >= board.x_size);
  testAssert(nnYLen >= board.y_size);
  testAssert(nnXLen > 0 && nnXLen < 100); //Just a sanity check to make sure no other crazy values have snuck in
  testAssert(nnYLen > 0 && nnYLen < 100); //Just a sanity check to make sure no other crazy values have snuck in
  int policySize = hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS : NNPos::getPolicySize(nnXLen,nnYLen);
  testAssert(pla == hist.presumedNextMovePla);
  for(int movePos = 0; movePos<policySize; movePos++) {
    Loc moveLoc = hist.rules.isXiangqi() ? Xiangqi::actionToLoc(movePos) :
      NNPos::posToLoc(movePos,board.x_size,board.y_size,nnXLen,nnYLen);
    double policyProb = nnOutput->policyProbs[movePos];
    if(!hist.isLegal(board,moveLoc,pla) || policyProb <= 0)
      continue;
    locs.push_back(moveLoc);
    playSelectionValues.push_back(pow(policyProb,1.0/temperature));
  }

  //In practice, this should never happen, but in theory, a very badly-behaved net that rounds
  //all legal moves to zero could result in this. We still go ahead and fail, since this more likely some sort of bug.
  if(playSelectionValues.size() <= 0)
    throw StringError("getGameInitializationMove: playSelectionValues.size() <= 0");

  //With a tiny probability, choose a uniformly random move instead of a policy move, to also
  //add a bit more outlierish variety
  uint32_t idxChosen;
  if(gameRand.nextBool(0.0002))
    idxChosen = gameRand.nextUInt((uint32_t)playSelectionValues.size());
  else
    idxChosen = gameRand.nextUInt(playSelectionValues.data(),playSelectionValues.size());
  Loc loc = locs[idxChosen];
  return loc;
}


//Try playing a bunch of pure policy moves instead of playing from the start to initialize the board
//and add entropy
int PlayUtils::samplePolicyInitMoveCount(
  const BoardHistory& hist, Rand& gameRand, double mean, double gammaShape
) {
  if(hist.rules.isPlacement() && hist.rules.game != Rules::Game::HEX) {
    // KataGomo's board.movenum counts played moves, not setup stones. Imported
    // positions retain earlier played turns in initialTurnNumber.
    const double played = (double)hist.initialTurnNumber + hist.moveHistory.size();
    return std::max(0,(int)floor(gameRand.nextExponential() * mean - 2.0 * played));
  }
  return gammaShape != 1.0 ? (int)floor(gameRand.nextGamma(gammaShape) * (mean/gammaShape)) :
    (int)floor(gameRand.nextExponential() * mean);
}

void PlayUtils::initializeGameUsingPolicy(
  Search* botB, Search* botW, Board& board, BoardHistory& hist, Player& pla,
  Rand& gameRand, bool doEndGameIfAllPassAlive,
  double mean, double policyInitGammaShape, double temperature
) {
  NNResultBuf buf;

  if(hist.isGameFinished)
    return;

  int numInitialMovesToPlay = samplePolicyInitMoveCount(hist,gameRand,mean,policyInitGammaShape);

  testAssert(numInitialMovesToPlay >= 0);
  for(int i = 0; i<numInitialMovesToPlay; i++) {
    Loc loc = getGameInitializationMove(botB, botW, board, hist, pla, buf, gameRand, temperature);

    //Make the move!
    assert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    pla = getOpp(pla);

    //Rarely, playing the random moves out this way will end the game
    if(doEndGameIfAllPassAlive)
      hist.endGameIfAllPassAlive(board);
    if(hist.isGameFinished)
      break;
  }
}


//Place black handicap stones, free placement
//Does NOT switch the initial player of the board history to white
void PlayUtils::playExtraBlack(
  Search* bot,
  int numExtraBlack,
  Board& board,
  BoardHistory& hist,
  double temperature,
  Rand& gameRand
) {
  Player pla = P_BLACK;

  if(!hist.isGameFinished) {
    NNResultBuf buf;
    bool botPassAliveMode = bot->mainEvaluatorHistoryModes().alwaysComputePassAliveUnderSuicideRules;
    bool botExcludeTerritoryAdjAtari = bot->mainEvaluatorHistoryModes().excludeTerritoryAdjacentToAtari;
    for(int i = 0; i<numExtraBlack; i++) {
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = bot->searchParams.drawEquivalentWinsForWhite;
      //Featurize the way this bot's own searches would, even if the passed history differs.
      nnInputParams.passAliveSuicideRulesOverride = botPassAliveMode ? 1 : 0;
      nnInputParams.excludeTerritoryAdjAtariOverride = botExcludeTerritoryAdjAtari ? 1 : 0;
      bot->evaluatePosition(board,hist,pla,nnInputParams,buf,false);
      std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);

      bool allowPass = false;
      Loc banMove = Board::NULL_LOC;
      Loc loc = chooseRandomPolicyMove(nnOutput.get(), board, hist, pla, gameRand, temperature, allowPass, banMove);
      if(loc == Board::NULL_LOC)
        break;

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      hist.clear(board,pla,hist.rules,0);
    }
  }

  bot->setPosition(pla,board,hist);
}

void PlayUtils::placeFixedHandicap(Board& board, int n) {
  int xSize = board.x_size;
  int ySize = board.y_size;
  if(xSize < 7 || ySize < 7)
    throw StringError("Board is too small for fixed handicap");
  if((xSize % 2 == 0 || ySize % 2 == 0) && n > 4)
    throw StringError("Fixed handicap > 4 is not allowed on boards with even dimensions");
  if((xSize <= 7 || ySize <= 7) && n > 4)
    throw StringError("Fixed handicap > 4 is not allowed on boards with size 7");
  if(n < 2)
    throw StringError("Fixed handicap < 2 is not allowed");
  if(n > 9)
    throw StringError("Fixed handicap > 9 is not allowed");

  board = Board(xSize,ySize);

  int xCoords[3]; //Corner, corner, side
  int yCoords[3]; //Corner, corner, side
  if(xSize <= 12) { xCoords[0] = 2; xCoords[1] = xSize-3; xCoords[2] = xSize/2; }
  else            { xCoords[0] = 3; xCoords[1] = xSize-4; xCoords[2] = xSize/2; }
  if(ySize <= 12) { yCoords[0] = 2; yCoords[1] = ySize-3; yCoords[2] = ySize/2; }
  else            { yCoords[0] = 3; yCoords[1] = ySize-4; yCoords[2] = ySize/2; }

  auto s = [&](int xi, int yi) {
    board.setStone(Location::getLoc(xCoords[xi],yCoords[yi],board.x_size),P_BLACK);
  };
  if(n == 2) { s(0,1); s(1,0); }
  else if(n == 3) { s(0,1); s(1,0); s(0,0); }
  else if(n == 4) { s(0,1); s(1,0); s(0,0); s(1,1); }
  else if(n == 5) { s(0,1); s(1,0); s(0,0); s(1,1); s(2,2); }
  else if(n == 6) { s(0,1); s(1,0); s(0,0); s(1,1); s(0,2); s(1,2); }
  else if(n == 7) { s(0,1); s(1,0); s(0,0); s(1,1); s(0,2); s(1,2); s(2,2); }
  else if(n == 8) { s(0,1); s(1,0); s(0,0); s(1,1); s(0,2); s(1,2); s(2,0); s(2,1); }
  else if(n == 9) { s(0,1); s(1,0); s(0,0); s(1,1); s(0,2); s(1,2); s(2,0); s(2,1); s(2,2); }
  else { ASSERT_UNREACHABLE; }
}

double PlayUtils::getHackedLCBForWinrate(const Search* search, const AnalysisData& data, Player pla) {
  double winrate = 0.5 * (1.0 + data.winLossValue);
  //Expected points span [0,1], half the [-1,1] result utility range.
  double radiusScaleHackFactor = 0.5 * search->searchParams.winLossUtilityFactor /
    (search->searchParams.winLossUtilityFactor + 1e-20);
  double lcb = pla == P_WHITE ? winrate - data.radius * radiusScaleHackFactor : winrate + data.radius * radiusScaleHackFactor;
  return lcb;
}

double PlayUtils::getSearchFactor(
  double searchFactorWhenWinningThreshold,
  double searchFactorWhenWinning,
  const SearchParams& params,
  const vector<double>& recentWinLossValues,
  Player pla
) {
  double searchFactor = 1.0;
  if(recentWinLossValues.size() >= 3 && params.winLossUtilityFactor - searchFactorWhenWinningThreshold > 1e-10) {
    double recentLeastWinning = pla == P_BLACK ? -params.winLossUtilityFactor : params.winLossUtilityFactor;
    for(size_t i = recentWinLossValues.size()-3; i < recentWinLossValues.size(); i++) {
      if(pla == P_BLACK && recentWinLossValues[i] > recentLeastWinning)
        recentLeastWinning = recentWinLossValues[i];
      if(pla == P_WHITE && recentWinLossValues[i] < recentLeastWinning)
        recentLeastWinning = recentWinLossValues[i];
    }
    double excessWinning = pla == P_BLACK ? -searchFactorWhenWinningThreshold - recentLeastWinning : recentLeastWinning - searchFactorWhenWinningThreshold;
    if(excessWinning > 0) {
      double lambda = excessWinning / (params.winLossUtilityFactor - searchFactorWhenWinningThreshold);
      searchFactor = 1.0 + lambda * (searchFactorWhenWinning - 1.0);
    }
  }
  return searchFactor;
}



//Tromp-taylor-like scoring, except recognizes pass-dead stones.
vector<bool> PlayUtils::computeAnticipatedStatusesSimple(
  const Board& board,
  const BoardHistory& hist
) {
  vector<bool> isAlive(Board::MAX_ARR_SIZE,false);

  //Treat all stones as alive under a no result
  if(!hist.rules.isGo() || (hist.isGameFinished && hist.isNoResult)) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY)
          isAlive[loc] = true;
      }
    }
  }
  //Else use Tromp-taylorlike scoring, except recognizing pass-dead stones.
  else {
    Color area[Board::MAX_ARR_SIZE];
    BoardHistory histCopy = hist;
    histCopy.endAndScoreGameNow(board,area);
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        Loc loc = Location::getLoc(x,y,board.x_size);
        if(board.colors[loc] != C_EMPTY) {
          isAlive[loc] = board.colors[loc] == area[loc];
        }
      }
    }
  }
  return isAlive;
}

string PlayUtils::BenchmarkResults::toStringNotDone() const {
  ostringstream out;
  out << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
      << " " << totalPositionsSearched << " / " << totalPositions << " positions,"
      << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
      << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";
  return out.str();
}
string PlayUtils::BenchmarkResults::toString() const {
  ostringstream out;
  out << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
      << " " << totalPositionsSearched << " / " << totalPositions << " positions,"
      << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
      << " nnEvals/s = " << Global::strprintf("%.2f",numNNEvals / totalSeconds)
      << " nnBatches/s = " << Global::strprintf("%.2f",numNNBatches / totalSeconds)
      << " avgBatchSize = " << Global::strprintf("%.2f",avgBatchSize)
      << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";
  return out.str();
}
string PlayUtils::BenchmarkResults::toStringWithElo(const BenchmarkResults* baseline, double secondsPerGameMove) const {
  ostringstream out;
  out << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ":"
      << " " << totalPositionsSearched << " / " << totalPositions << " positions,"
      << " visits/s = " << Global::strprintf("%.2f",totalVisits / totalSeconds)
      << " nnEvals/s = " << Global::strprintf("%.2f",numNNEvals / totalSeconds)
      << " nnBatches/s = " << Global::strprintf("%.2f",numNNBatches / totalSeconds)
      << " avgBatchSize = " << Global::strprintf("%.2f",avgBatchSize)
      << " (" << Global::strprintf("%.1f", totalSeconds) << " secs)";

  if(baseline == NULL)
    out << " (EloDiff baseline)";
  else {
    double diff = computeEloEffect(secondsPerGameMove) - baseline->computeEloEffect(secondsPerGameMove);
    out << " (EloDiff " << Global::strprintf("%+.0f",diff) << ")";
  }
  return out.str();
}

//From some test matches by lightvector using g170
static constexpr double eloGainPerDoubling = 250;

double PlayUtils::BenchmarkResults::computeEloEffect(double secondsPerGameMove) const {
  auto computeEloCost = [&](double baseVisits) {
    //Completely ad-hoc formula that approximately fits noisy tests. Probably not very good
    //but then again the recommendation of this benchmark program is very rough anyways, it
    //doesn't need to be all that great.
    return numThreads * 7.0 * pow(1600.0 / (800.0 + baseVisits),0.85);
  };

  double visitsPerSecond = totalVisits / totalSeconds;
  double gain = eloGainPerDoubling * log(visitsPerSecond) / log(2);
  double visitsPerMove = visitsPerSecond * secondsPerGameMove;
  double cost = computeEloCost(visitsPerMove);
  return gain - cost;
}

void PlayUtils::BenchmarkResults::printEloComparison(const vector<BenchmarkResults>& results, double secondsPerGameMove) {
  int bestIdx = 0;
  for(int i = 1; i<results.size(); i++) {
    if(results[i].computeEloEffect(secondsPerGameMove) > results[bestIdx].computeEloEffect(secondsPerGameMove))
      bestIdx = i;
  }

  cout << endl;
  cout << "Based on some test data, each speed doubling gains perhaps ~" << eloGainPerDoubling << " Elo by searching deeper." << endl;
  cout << "Based on some test data, each thread costs perhaps 7 Elo if using 800 visits, and 2 Elo if using 5000 visits (by making MCTS worse)." << endl;
  cout << "So APPROXIMATELY based on this benchmark, if you intend to do a " << secondsPerGameMove << " second search: " << endl;
  for(int i = 0; i<results.size(); i++) {
    int numThreads = results[i].numThreads;
    double eloEffect = results[i].computeEloEffect(secondsPerGameMove) - results[0].computeEloEffect(secondsPerGameMove);
    cout << "numSearchThreads = " << Global::strprintf("%2d",numThreads) << ": ";
    if(i == 0)
      cout << "(baseline)" << (i == bestIdx ? " (recommended)" : "") << endl;
    else
      cout << Global::strprintf("%+5.0f",eloEffect) << " Elo" << (i == bestIdx ? " (recommended)" : "") << endl;
  }
  cout << endl;
}


PlayUtils::BenchmarkResults PlayUtils::benchmarkSearchOnPositionsAndPrint(
  const SearchParams& params,
  const CompactSgf& sgf,
  int numPositionsToUse,
  NNEvaluator* nnEval,
  const BenchmarkResults* baseline,
  double secondsPerGameMove,
  bool printElo,
  SelfplayModel* model
) {
  //Pick random positions from the SGF file, but deterministically
  vector<Move> moves = sgf.moves;
  if(moves.size() > 0xFFFF)
    moves.resize(0xFFFF);
  string posSeed = "benchmarkPosSeed|";
  for(int i = 0; i<moves.size(); i++) {
    posSeed += Global::intToString((int)moves[i].loc);
    posSeed += "|";
  }

  vector<int> possiblePositionIdxs;
  {
    Rand posRand(posSeed);
    for(int i = 0; i<moves.size(); i++) {
      possiblePositionIdxs.push_back(i);
    }
    if(possiblePositionIdxs.size() > 0) {
      for(int i = (int)possiblePositionIdxs.size()-1; i > 1; i--) {
        int r = posRand.nextUInt(i);
        int tmp = possiblePositionIdxs[i];
        possiblePositionIdxs[i] = possiblePositionIdxs[r];
        possiblePositionIdxs[r] = tmp;
      }
    }
    if(possiblePositionIdxs.size() > numPositionsToUse)
      possiblePositionIdxs.resize(numPositionsToUse);
  }

  std::sort(possiblePositionIdxs.begin(),possiblePositionIdxs.end());

  BenchmarkResults results;
  results.numThreads = params.numThreads;
  results.totalPositions = (int)possiblePositionIdxs.size();

  if(model) { model->clearCache(); model->clearStats(); }
  else { nnEval->clearCache(); nnEval->clearStats(); }

  Rand seedRand;
  MuZero::PositionEvaluator* muZeroEval = model ? model->muZeroEvaluator() : nullptr;
  Search* bot = muZeroEval
    ? new Search(params,*muZeroEval,model->getLogger(),Global::uint64ToString(seedRand.nextUInt64()))
    : new Search(params,nnEval,nnEval->getLogger(),Global::uint64ToString(seedRand.nextUInt64()));

  Rules initialRules = sgf.getRulesOrFailAllowUnspecified(Rules::getTrompTaylorish());
  // Retain source Go benchmarking rules. Other games need their actual rules
  // and Xiangqi history rather than a Go interpretation of their coordinates.
  if(initialRules.isGo()) {
    float komi = initialRules.komi;
    initialRules = Rules::getTrompTaylorish();
    initialRules.komi = komi;
  }

  Board board;
  Player nextPla;
  BoardHistory hist;
  sgf.setupInitialBoardAndHist(initialRules, board, nextPla, hist,
    muZeroEval ? Search::resolveHistoryModes(params,*muZeroEval) : Search::resolveHistoryModes(params,nnEval));

  int moveNum = 0;

  for(int i = 0; i<possiblePositionIdxs.size(); i++) {
    cout << "\r" << results.toStringNotDone() << "      " << std::flush;

    int nextIdx = possiblePositionIdxs[i];
    while(moveNum < moves.size() && moveNum < nextIdx) {
      bool suc = hist.makeBoardMoveTolerant(board,moves[moveNum].loc,moves[moveNum].pla);
      if(!suc) {
        cerr << endl;
        cerr << board << endl;
        cerr << "SGF Illegal move " << (moveNum+1) << " for " << PlayerIO::colorToChar(moves[moveNum].pla) << ": " <<
          (hist.rules.isXiangqi() ? Xiangqi::actionToString(Xiangqi::locToAction(moves[moveNum].loc)) : Location::toString(moves[moveNum].loc,board)) << endl;
        throw StringError("Illegal move in SGF");
      }
      nextPla = getOpp(moves[moveNum].pla);
      moveNum += 1;
    }

    bot->clearSearch();
    bot->setPosition(nextPla,board,hist);
    if(model) model->clearCache();
    else nnEval->clearCache();

    ClockTimer timer;
    bot->runWholeSearch(nextPla);
    double seconds = timer.getSeconds();

    results.totalPositionsSearched += 1;
    results.totalSeconds += seconds;
    results.totalVisits += bot->getRootVisits();
  }

  results.numNNEvals = model ? model->numRowsProcessed() : nnEval->numRowsProcessed();
  results.numNNBatches = model ? model->numBatchesProcessed() : nnEval->numBatchesProcessed();
  results.avgBatchSize = model ? model->averageProcessedBatchSize() : nnEval->averageProcessedBatchSize();

  if(printElo)
    cout << "\r" << results.toStringWithElo(baseline,secondsPerGameMove) << std::endl;
  else
    cout << "\r" << results.toString() << std::endl;

  delete bot;

  return results;
}


void PlayUtils::printGenmoveLog(
  ostream& out,
  const Search* search,
  const NNEvaluator* nnEval,
  Loc moveLoc,
  double timeTaken,
  Player perspective,
  bool logSearchInfoForChosenMove,
  const PrintTreeOptions& treeOptions,
  const SelfplayModel* model
) {
  const Board& board = search->getRootBoard();
  if(search->getRootHist().rules.isXiangqi()) search->getRootHist().printBasicInfo(out,board);
  else Board::printBoard(out, board, moveLoc, &(search->getRootHist().moveHistory));
  out << search->getRootHist().rules << "\n";
  if(!std::isnan(timeTaken))
    out << "Time taken: " << timeTaken << "\n";
  out << "Root visits: " << search->getRootVisits() << "\n";
  out << "New playouts: " << search->lastSearchNumPlayouts << "\n";
  out << "NN rows: " << (nnEval ? nnEval->numRowsProcessed() : model->numRowsProcessed()) << endl;
  out << "NN batches: " << (nnEval ? nnEval->numBatchesProcessed() : model->numBatchesProcessed()) << endl;
  out << "NN avg batch size: " << (nnEval ? nnEval->averageProcessedBatchSize() : model->averageProcessedBatchSize()) << endl;
  if(search->searchParams.playoutDoublingAdvantage != 0)
    out << "PlayoutDoublingAdvantage: " << (
      search->getRootPla() == getOpp(search->getPlayoutDoublingAdvantagePla()) ?
      -search->searchParams.playoutDoublingAdvantage : search->searchParams.playoutDoublingAdvantage) << endl;
  if(search->searchParams.visitCapContempt > 0)
    out << "VisitCapContempt: " << search->searchParams.visitCapContempt
        << " (capped player " << PlayerIO::playerToString(search->getVisitCappedPla()) << ")" << endl;
  out << "PV: ";
  search->printPV(out, search->rootNode, 25);
  out << "\n";
  out << "Tree:\n";
  if(logSearchInfoForChosenMove && moveLoc != Board::NULL_LOC) {
    if(search->getRootHist().rules.isXiangqi()) {
      PrintTreeOptions options = treeOptions;
      options.branch_ = {moveLoc};
      options.alsoBranch_ = true;
      search->printTree(out,search->rootNode,options,perspective);
    }
    else search->printTree(out, search->rootNode, treeOptions.alsoBranch(board,{Location::toString(moveLoc,board)}),perspective);
  }
  else
    search->printTree(out, search->rootNode, treeOptions,perspective);
}

Rules PlayUtils::genRandomRules(Rand& rand) {
  vector<int> allowedKoRules = { Rules::KO_SIMPLE, Rules::KO_POSITIONAL, Rules::KO_SITUATIONAL };
  vector<int> allowedScoringRules = { Rules::SCORING_AREA, Rules::SCORING_TERRITORY };
  vector<int> allowedTaxRules = { Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL };

  Rules rules;
  rules.koRule = allowedKoRules[rand.nextUInt((uint32_t)allowedKoRules.size())];
  rules.scoringRule = allowedScoringRules[rand.nextUInt((uint32_t)allowedScoringRules.size())];
  rules.taxRule = allowedTaxRules[rand.nextUInt((uint32_t)allowedTaxRules.size())];
  rules.multiStoneSuicideLegal = rand.nextBool(0.5);

  if(rules.scoringRule == Rules::SCORING_AREA)
    rules.hasButton = rand.nextBool(0.5);
  else
    rules.hasButton = false;
  return rules;
}

std::shared_ptr<NNOutput> PlayUtils::getFullSymmetryNNOutput(
  const Board& board, const BoardHistory& hist, Player pla, const SGFMetadata* sgfMeta, NNEvaluator* nnEval
) {
  vector<std::shared_ptr<NNOutput>> ptrs;
  for(int sym = 0; sym<SymmetryHelpers::NUM_SYMMETRIES; sym++) {
    if(!hist.rules.allowsSymmetry(sym)) continue;
    MiscNNInputParams nnInputParams;
    nnInputParams.symmetry = sym;
    NNResultBuf buf;
    bool skipCache = true; //Always ignore cache so that we use the desired symmetry
    nnEval->evaluate(board,hist,pla,sgfMeta,nnInputParams,buf,skipCache);
    ptrs.push_back(std::move(buf.result));
  }
  std::shared_ptr<NNOutput> result(new NNOutput(ptrs));
  return result;
}
