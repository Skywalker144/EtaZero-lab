#include "../program/play.h"

#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/timer.h"
#include "../program/playutils.h"
#include "../program/setup.h"
#include "../search/asyncbot.h"
#include "../search/searchnode.h"
#include "../search/policybuffer.h"
#include "../game/xiangqi.h"
#include "../dataio/files.h"

#include "../core/test.h"

using namespace std;

static Hash128 hintPositionHash(const Board& board, const BoardHistory& hist) {
  return hist.rules.isXiangqi() ? hist.getXiangqiState().historyHash() : board.pos_hash;
}

//----------------------------------------------------------------------------------------------------------

InitialPosition::InitialPosition()
  :board(),hist(),pla(C_EMPTY)
{}
InitialPosition::InitialPosition(const Board& b, const BoardHistory& h, Player p, bool plainFork, bool hintFork, double tw)
  :board(b),hist(h),pla(p),isPlainFork(plainFork),isHintFork(hintFork),trainingWeight(tw)
{}
InitialPosition::~InitialPosition()
{}


ForkData::~ForkData() {
  for(int i = 0; i<forks.size(); i++)
    delete forks[i];
  forks.clear();
}

void ForkData::add(const InitialPosition* pos) {
  std::lock_guard<std::mutex> lock(mutex);
  forks.push_back(pos);
}
const InitialPosition* ForkData::get(Rand& rand) {
  std::lock_guard<std::mutex> lock(mutex);
  if(forks.size() <= 0)
    return NULL;
  testAssert(forks.size() < 0x1FFFffff);
  uint32_t r = rand.nextUInt((uint32_t)forks.size());
  size_t last = forks.size()-1;
  const InitialPosition* pos = forks[r];
  forks[r] = forks[last];
  forks.resize(forks.size()-1);
  return pos;
}

//------------------------------------------------------------------------------------------------

GameInitializer::GameInitializer(ConfigParser& cfg, Logger& logger)
  :createGameMutex(),rand()
{
  initShared(cfg,logger);
}

GameInitializer::GameInitializer(ConfigParser& cfg, Logger& logger, const string& randSeed)
  :createGameMutex(),rand(randSeed)
{
  initShared(cfg,logger);
}

void GameInitializer::initShared(ConfigParser& cfg, Logger& logger) {

  game = cfg.contains("game") ? Rules::parseGame(cfg.getString("game")) : Rules::Game::GO;
  if(game == Rules::Game::GO) {
    allowedKoRuleStrs = cfg.getStrings("koRules", Rules::koRuleStrings());
    allowedScoringRuleStrs = cfg.getStrings("scoringRules", Rules::scoringRuleStrings());
    allowedTaxRuleStrs = cfg.getStrings("taxRules", Rules::taxRuleStrings());
    allowedMultiStoneSuicideLegals = cfg.getBools("multiStoneSuicideLegals");
    allowedButtons = cfg.getBools("hasButtons");

    for(size_t i = 0; i < allowedKoRuleStrs.size(); i++)
      allowedKoRules.push_back(Rules::parseKoRule(allowedKoRuleStrs[i]));
    for(size_t i = 0; i < allowedScoringRuleStrs.size(); i++)
      allowedScoringRules.push_back(Rules::parseScoringRule(allowedScoringRuleStrs[i]));
    for(size_t i = 0; i < allowedTaxRuleStrs.size(); i++)
      allowedTaxRules.push_back(Rules::parseTaxRule(allowedTaxRuleStrs[i]));

    if(allowedKoRules.size() <= 0)
      throw IOError("koRules must have at least one value in " + cfg.getFileName());
    if(allowedScoringRules.size() <= 0)
      throw IOError("scoringRules must have at least one value in " + cfg.getFileName());
    if(allowedTaxRules.size() <= 0)
      throw IOError("taxRules must have at least one value in " + cfg.getFileName());
    if(allowedMultiStoneSuicideLegals.size() <= 0)
      throw IOError("multiStoneSuicideLegals must have at least one value in " + cfg.getFileName());
    if(allowedButtons.size() <= 0)
      throw IOError("hasButtons must have at least one value in " + cfg.getFileName());

    {
      bool hasAreaScoring = false;
      for(int i = 0; i<allowedScoringRules.size(); i++)
        if(allowedScoringRules[i] == Rules::SCORING_AREA)
          hasAreaScoring = true;
      bool hasTrueButton = false;
      for(int i = 0; i<allowedButtons.size(); i++)
        if(allowedButtons[i])
          hasTrueButton = true;
      if(!hasAreaScoring && hasTrueButton)
        throw IOError("If scoringRules does not include AREA, hasButtons must be false in " + cfg.getFileName());
    }

  }

  if(cfg.contains("bSizes") == cfg.contains("bSizesXY"))
    throw IOError("Must specify exactly one of bSizes or bSizesXY");

  if(cfg.contains("bSizes")) {
    std::vector<int> allowedBEdges = cfg.getInts("bSizes", 2, Board::MAX_LEN);
    std::vector<double> allowedBEdgeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1e100);
    double relProbSum = 0.0;
    for(const double p : allowedBEdgeRelProbs)
      relProbSum += p;
    if(relProbSum <= 1e-100)
      throw IOError("bSizeRelProbs must sum to a positive value");
    double allowRectangleProb = cfg.contains("allowRectangleProb") ? cfg.getDouble("allowRectangleProb",0.0,1.0) : 0.0;

    if(allowedBEdges.size() <= 0)
      throw IOError("bSizes must have at least one value in " + cfg.getFileName());
    if(allowedBEdges.size() != allowedBEdgeRelProbs.size())
      throw IOError("bSizes and bSizeRelProbs must have same number of values in " + cfg.getFileName());

    allowedBSizes.clear();
    allowedBSizeRelProbs.clear();
    for(int i = 0; i<(int)allowedBEdges.size(); i++) {
      for(int j = 0; j<(int)allowedBEdges.size(); j++) {
        int x = allowedBEdges[i];
        int y = allowedBEdges[j];
        if(x == y) {
          allowedBSizes.emplace_back(x,y);
          allowedBSizeRelProbs.push_back(
            (1.0 - allowRectangleProb) * allowedBEdgeRelProbs[i] / relProbSum +
            allowRectangleProb * allowedBEdgeRelProbs[i] * allowedBEdgeRelProbs[j] / relProbSum / relProbSum
          );
        }
        else {
          if(allowRectangleProb > 0.0) {
            allowedBSizes.emplace_back(x,y);
            allowedBSizeRelProbs.push_back(
              allowRectangleProb * allowedBEdgeRelProbs[i] * allowedBEdgeRelProbs[j] / relProbSum / relProbSum
            );
          }
        }
      }
    }
  }
  else if(cfg.contains("bSizesXY")) {
    if(cfg.contains("allowRectangleProb"))
      throw IOError("Cannot specify allowRectangleProb when specifying bSizesXY, please adjust the relative frequency of rectangles yourself");
    allowedBSizes = cfg.getNonNegativeIntDashedPairs("bSizesXY", 2, Board::MAX_LEN);
    allowedBSizeRelProbs = cfg.getDoubles("bSizeRelProbs",0.0,1e100);

    double relProbSum = 0.0;
    for(const double p : allowedBSizeRelProbs)
      relProbSum += p;
    if(relProbSum <= 1e-100)
      throw IOError("bSizeRelProbs must sum to a positive value");
  }

  for(const char* key : {"komiStdev", "komiAllowIntegerProb", "handicapCompensateKomiProb", "forkCompensateKomiProb", "sgfCompensateKomiProb", "komiBigStdevProb", "komiBigStdev", "komiBiggerStdevProb", "komiBiggerStdev", "handicapKomiInterpZeroProb", "sgfKomiInterpZeroProb", "komiAuto"}) {
    if(cfg.contains(key))
      throw IOError(string("EtaZero uses fixed rule komi; removed setting: ") + key);
  }
  komiMean = game == Rules::Game::GO ? cfg.getFloat("komiMean",Rules::MIN_USER_KOMI,Rules::MAX_USER_KOMI) :
    (cfg.contains("komiMean") ? cfg.getFloat("komiMean",0.0f,0.0f) : 0.0f);
  if(!Rules::komiIsIntOrHalfInt(komiMean))
    throw IOError("komiMean must be an integer or half-integer fixed rule komi");
  handicapProb = cfg.contains("handicapProb") ? cfg.getDouble("handicapProb",0.0,1.0) : 0.0;

  auto generateCumProbs = [](const vector<Sgf::PositionSample>& poses, double lambda, double& effectiveSampleSize) {
    int64_t minInitialTurnNumber = 0;
    for(size_t i = 0; i<poses.size(); i++)
      minInitialTurnNumber = std::min(minInitialTurnNumber, poses[i].initialTurnNumber);

    vector<double> cumProbs;
    cumProbs.resize(poses.size());
    // Fill with uncumulative probs
    for(size_t i = 0; i<poses.size(); i++) {
      int64_t startTurn = poses[i].getCurrentTurnNumber() - minInitialTurnNumber;
      cumProbs[i] = exp(-startTurn * lambda) * poses[i].weight;
    }
    for(size_t i = 0; i<poses.size(); i++) {
      if(!(cumProbs[i] > -1e200 && cumProbs[i] < 1e200)) {
        throw StringError("startPos found bad unnormalized probability: " + Global::doubleToString(cumProbs[i]));
      }
    }

    // Compute ESS
    double sum = 0.0;
    double sumSq = 0.0;
    for(size_t i = 0; i<poses.size(); i++) {
      sum += cumProbs[i];
      sumSq += cumProbs[i]*cumProbs[i];
    }
    effectiveSampleSize = sum * sum / (sumSq + 1e-200);

    // Make cumulative
    for(size_t i = 1; i<poses.size(); i++)
      cumProbs[i] += cumProbs[i-1];

    return cumProbs;
  };

  startPosesProb = 0.0;
  if(cfg.contains("startPosesFromSgfDir")) {
    startPoses.clear();
    startPosCumProbs.clear();
    startPosesProb = cfg.getDouble("startPosesProb",0.0,1.0);

    vector<string> dirs = Global::split(cfg.getString("startPosesFromSgfDir"),',');
    vector<string> excludes = Global::split(cfg.contains("startPosesSgfExcludes") ? cfg.getString("startPosesSgfExcludes") : "",',');
    double startPosesLoadProb = cfg.getDouble("startPosesLoadProb",0.0,1.0);
    double startPosesTurnWeightLambda = cfg.getDouble("startPosesTurnWeightLambda",-10,10);

    vector<string> files;
    FileHelpers::collectSgfsFromDirs(dirs,files);
    std::set<Hash128> excludeHashes = Sgf::readExcludes(excludes);
    logger.write("Found " + Global::uint64ToString(files.size()) + " sgf files");
    logger.write("Loaded " + Global::uint64ToString(excludeHashes.size()) + " excludes");
    std::set<Hash128> uniqueHashes;
    std::function<void(Sgf::PositionSample&, const BoardHistory&, const string&)> posHandler = [startPosesLoadProb,this](
      const Sgf::PositionSample& posSample, const BoardHistory& hist, const string& comments
    ) {
      (void)hist;
      (void)comments;
      if(rand.nextBool(startPosesLoadProb))
        startPoses.push_back(posSample);
    };
    int64_t numExcluded = 0;
    for(size_t i = 0; i<files.size(); i++) {
      std::unique_ptr<Sgf> sgf = nullptr;
      try {
        sgf = Sgf::loadFile(files[i]);
        if(sgf->nodes[0]->getGame() != game)
          throw StringError("SGF game does not match configured game " + Rules::gameName(game));
        if(contains(excludeHashes,sgf->hash))
          numExcluded += 1;
        else {
          bool hashComments = false;
          bool hashParent = false;
          bool flipIfPassOrWFirst = true;
          bool allowGameOver = false;
          sgf->iterAllUniquePositions(uniqueHashes, hashComments, hashParent, flipIfPassOrWFirst, allowGameOver, NULL, posHandler);
        }
      }
      catch(const StringError& e) {
        logger.write("Invalid SGF " + files[i] + ": " + e.what());
      }
    }
    logger.write("Kept " + Global::uint64ToString(startPoses.size()) + " start positions");
    logger.write("Excluded " + Global::int64ToString(numExcluded) + "/" + Global::uint64ToString(files.size()) + " sgf files");

    double ess = 0.0;
    startPosCumProbs = generateCumProbs(startPoses, startPosesTurnWeightLambda, ess);

    if(startPoses.size() <= 0) {
      logger.write("No start positions loaded, disabling start position logic");
      startPosesProb = 0;
    }
    else {
      logger.write("Cumulative unnormalized probability for start poses: " + Global::doubleToString(startPosCumProbs[startPoses.size()-1]));
      logger.write("Effective sample size for start poses: " + Global::doubleToString(ess));
    }
  }

  hintPosesProb = 0.0;
  if(cfg.contains("hintPosesDir")) {
    hintPoses.clear();
    hintPosCumProbs.clear();
    hintPosesProb = cfg.getDouble("hintPosesProb",0.0,1.0);

    vector<string> dirs = Global::split(cfg.getString("hintPosesDir"),',');

    vector<string> files;
    std::function<bool(const string&)> fileFilter = [](const string& fileName) {
      return
        Global::isSuffix(fileName,".hintposes.txt") ||
        Global::isSuffix(fileName,".startposes.txt") ||
        Global::isSuffix(fileName,".bookposes.txt");
    };
    for(int i = 0; i<dirs.size(); i++) {
      string dir = Global::trim(dirs[i]);
      if(dir.size() > 0)
        FileUtils::collectFiles(dir, fileFilter, files);
    }

    for(size_t i = 0; i<files.size(); i++) {
      vector<string> lines = FileUtils::readFileLines(files[i],'\n');
      for(size_t j = 0; j<lines.size(); j++) {
        string line = Global::trim(lines[j]);
        if(line.size() > 0) {
          try {
            Sgf::PositionSample posSample = Sgf::PositionSample::ofJsonLine(line);
            if(posSample.game != game)
              throw StringError("Hint game does not match configured game " + Rules::gameName(game));
            if(game != Rules::Game::GO) {
              BoardHistory hist;
              Player pla;
              if(!posSample.tryGetCurrentBoardHistory(Rules::forGame(game),pla,hist,BoardHistoryModes()) || hist.isGameFinished)
                throw StringError("Placement hint must describe a legal nonterminal position");
              if(posSample.hintLoc != Board::NULL_LOC && !hist.isLegal(hist.getRecentBoard(0),posSample.hintLoc,pla))
                throw StringError("Illegal placement hint move");
            }
            hintPoses.push_back(posSample);
          }
          catch(const StringError& err) {
            logger.write(string("ERROR parsing hintpos: ") + err.what());
          }
        }
      }
    }
    logger.write("Loaded " + Global::uint64ToString(hintPoses.size()) + " hint positions");

    double ess = 0.0;
    hintPosCumProbs = generateCumProbs(hintPoses, 0.0, ess);

    if(hintPoses.size() <= 0) {
      logger.write("No hint positions loaded, disabling hint position logic");
      hintPosesProb = 0;
    }
    else {
      logger.write("Cumulative unnormalized probability for hint poses: " + Global::doubleToString(hintPosCumProbs[hintPoses.size()-1]));
      logger.write("Effective sample size for hint poses: " + Global::doubleToString(ess));
    }
  }

  if(allowedBSizes.size() <= 0)
    throw IOError("bSizes or bSizesXY must have at least one value in " + cfg.getFileName());
  if(allowedBSizes.size() != allowedBSizeRelProbs.size())
    throw IOError("bSizes or bSizesXY and bSizeRelProbs must have same number of values in " + cfg.getFileName());

  minBoardXSize = allowedBSizes[0].first;
  minBoardYSize = allowedBSizes[0].second;
  maxBoardXSize = allowedBSizes[0].first;
  maxBoardYSize = allowedBSizes[0].second;
  for(const std::pair<int,int>& bSize : allowedBSizes) {
    if(game == Rules::Game::XIANGQI && (bSize.first != Xiangqi::WIDTH || bSize.second != Xiangqi::HEIGHT))
      throw IOError("Xiangqi requires bSizesXY = 9-10");
    minBoardXSize = std::min(minBoardXSize, bSize.first);
    minBoardYSize = std::min(minBoardYSize, bSize.second);
    maxBoardXSize = std::max(maxBoardXSize, bSize.first);
    maxBoardYSize = std::max(maxBoardYSize, bSize.second);
  }
  for(const Sgf::PositionSample& pos : hintPoses) {
    minBoardXSize = std::min(minBoardXSize, pos.board.x_size);
    minBoardYSize = std::min(minBoardYSize, pos.board.y_size);
    maxBoardXSize = std::max(maxBoardXSize, pos.board.x_size);
    maxBoardYSize = std::max(maxBoardYSize, pos.board.y_size);
  }

  noResultStdev = cfg.contains("noResultStdev") ? cfg.getDouble("noResultStdev",0.0,1.0) : 0.0;
  numExtraBlackFixed = cfg.contains("numExtraBlackFixed") ? cfg.getInt("numExtraBlackFixed",1,18) : 0;
  if(cfg.contains("drawRandRadius")) cfg.getDouble("drawRandRadius",0.0,0.0);
  if(game != Rules::Game::GO && (handicapProb != 0 || numExtraBlackFixed != 0))
    throw IOError("Go handicap initialization cannot be used for non-Go games");
}

GameInitializer::~GameInitializer()
{}

void GameInitializer::createGame(
  Board& board, Player& pla, BoardHistory& hist,
  int& numExtraBlack,
  const InitialPosition* initialPosition,
  const PlaySettings& playSettings,
  OtherGameProperties& otherGameProps,
  const Sgf::PositionSample* startPosSample,
  const BoardHistoryModes& gameHistoryModes
) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  if(noResultStdev != 0.0)
    throw StringError("GameInitializer::createGame called in a mode that doesn't support specifying noResultStdev");
  createGameSharedUnsynchronized(board,pla,hist,numExtraBlack,initialPosition,playSettings,otherGameProps,startPosSample,gameHistoryModes);
}

void GameInitializer::createGame(
  Board& board, Player& pla, BoardHistory& hist,
  int& numExtraBlack,
  SearchParams& params,
  const InitialPosition* initialPosition,
  const PlaySettings& playSettings,
  OtherGameProperties& otherGameProps,
  const Sgf::PositionSample* startPosSample,
  const BoardHistoryModes& gameHistoryModes
) {
  //Multiple threads will be calling this, and we have some mutable state such as rand.
  lock_guard<std::mutex> lock(createGameMutex);
  createGameSharedUnsynchronized(board,pla,hist,numExtraBlack,initialPosition,playSettings,otherGameProps,startPosSample,gameHistoryModes);

  if(params.drawEquivalentWinsForWhite != 0.5)
    throw StringError("EtaZero draw utility is fixed at zero");
  if(noResultStdev > 1e-30) {
    double mean = params.noResultUtilityForWhite;
    params.noResultUtilityForWhite = mean + noResultStdev * rand.nextGaussianTruncated(3.0);
    while(params.noResultUtilityForWhite < -1.0 || params.noResultUtilityForWhite > 1.0)
      params.noResultUtilityForWhite = mean + noResultStdev * rand.nextGaussianTruncated(3.0);
  }
}


bool GameInitializer::isAllowedBSize(int xSize, int ySize) const {
  if(!contains(allowedBSizes,std::make_pair(xSize,ySize)))
    return false;
  return true;
}

std::vector<std::pair<int,int>> GameInitializer::getAllowedBSizes() const {
  return allowedBSizes;
}
int GameInitializer::getMinBoardXSize() const {
  return minBoardXSize;
}
int GameInitializer::getMinBoardYSize() const {
  return minBoardYSize;
}
int GameInitializer::getMaxBoardXSize() const {
  return maxBoardXSize;
}
int GameInitializer::getMaxBoardYSize() const {
  return maxBoardYSize;
}

Rules GameInitializer::createRules() {
  lock_guard<std::mutex> lock(createGameMutex);
  return createRulesUnsynchronized();
}

Rules GameInitializer::createRulesUnsynchronized() {
  if(game != Rules::Game::GO) return Rules::forGame(game);
  Rules rules;
  rules.komi = komiMean;
  rules.koRule = allowedKoRules[rand.nextUInt((uint32_t)allowedKoRules.size())];
  rules.scoringRule = allowedScoringRules[rand.nextUInt((uint32_t)allowedScoringRules.size())];
  rules.taxRule = allowedTaxRules[rand.nextUInt((uint32_t)allowedTaxRules.size())];
  rules.multiStoneSuicideLegal = allowedMultiStoneSuicideLegals[rand.nextUInt((uint32_t)allowedMultiStoneSuicideLegals.size())];

  if(rules.scoringRule == Rules::SCORING_AREA)
    rules.hasButton = allowedButtons[rand.nextUInt((uint32_t)allowedButtons.size())];
  else
    rules.hasButton = false;
  return rules;
}

void GameInitializer::createGameSharedUnsynchronized(
  Board& board, Player& pla, BoardHistory& hist,
  int& numExtraBlack,
  const InitialPosition* initialPosition,
  const PlaySettings& playSettings,
  OtherGameProperties& otherGameProps,
  const Sgf::PositionSample* startPosSample,
  const BoardHistoryModes& gameHistoryModes
) {
  //The game-level BoardHistoryModes, so that any adjudication during start position
  //replay below is consistent with the game that will be played. (hist.clear() preserves these.)
  hist.setModes(gameHistoryModes);

  if(initialPosition != NULL) {
    if(initialPosition->hist.rules.game != game) throw StringError("Fork game does not match the configured game");
    board = initialPosition->board;
    hist = initialPosition->hist;
    hist.setModes(gameHistoryModes);
    pla = initialPosition->pla;

    numExtraBlack = 0;
    otherGameProps.isSgfPos = false;
    otherGameProps.isHintPos = false;
    otherGameProps.allowPolicyInit = false; //On fork positions, don't play extra moves at start
    otherGameProps.isFork = true;
    otherGameProps.isHintFork = initialPosition->isHintFork;
    otherGameProps.hintLoc = Board::NULL_LOC;
    otherGameProps.hintTurn = initialPosition->isHintFork ? (int)hist.moveHistory.size() : -1;
    otherGameProps.trainingWeight = initialPosition->trainingWeight;
    return;
  }

  int bSizeIdx = rand.nextUInt(allowedBSizeRelProbs.data(),allowedBSizeRelProbs.size());
  Rules rules = createRulesUnsynchronized();

  const Sgf::PositionSample* posSample = NULL;
  if(startPosSample != NULL) {
    posSample = startPosSample;
  }

  if(posSample == NULL) {
    if(startPosesProb > 0 && rand.nextBool(startPosesProb)) {
      testAssert(startPoses.size() > 0);
      size_t r = rand.nextIndexCumulative(startPosCumProbs.data(),startPosCumProbs.size());
      assert(r < startPosCumProbs.size());
      posSample = &(startPoses[r]);
    }
    else if(hintPosesProb > 0 && rand.nextBool(hintPosesProb)) {
      testAssert(hintPoses.size() > 0);
      size_t r = rand.nextIndexCumulative(hintPosCumProbs.data(),hintPosCumProbs.size());
      assert(r < hintPosCumProbs.size());
      posSample = &(hintPoses[r]);
    }
  }

  if(posSample != NULL) {
    const Sgf::PositionSample& startPos = *posSample;
    if(startPos.game != game) throw StringError("Position sample game does not match configured game");
    board = startPos.board;
    pla = startPos.nextPla;
    if(rules.isXiangqi()) {
      if(!startPos.xiangqiState || startPos.xiangqiState->nextPlayer() != pla)
        throw StringError("Xiangqi start position requires its complete state and matching player");
      hist.clear(*startPos.xiangqiState);
      board = hist.getRecentBoard(0);
    }
    else hist.clear(board,pla,rules,0);
    hist.setInitialTurnNumber(startPos.initialTurnNumber);
    Loc hintLoc = startPos.hintLoc;
    testAssert(startPos.moves.size() < 0xFFFFFF);
    for(size_t i = 0; i<startPos.moves.size(); i++) {
      bool isLegal = hist.isLegal(board,startPos.moves[i].loc,startPos.moves[i].pla);
      //Makes a best effort to still use the position, stopping if we hit an illegal move. It's possible
      //we hit this because of rules differences making a superko move or self-capture illegal, for example.
      if(!isLegal || hist.isGameFinished) {
        if(!rules.isGo()) throw StringError("Illegal move in placement start position");
        //If we stop due to illegality, it doesn't make sense to still use the hintLoc
        hintLoc = Board::NULL_LOC;
        break;
      }
      hist.makeBoardMoveAssumeLegal(board,startPos.moves[i].loc,startPos.moves[i].pla,NULL);
      pla = getOpp(startPos.moves[i].pla);
    }

    if(!rules.isGo() && (hist.isGameFinished || (hintLoc != Board::NULL_LOC && !hist.isLegal(board,hintLoc,pla))))
      throw StringError("Placement start position is terminal or has an illegal hint");
    numExtraBlack = 0;

    otherGameProps.isSgfPos = hintLoc == Board::NULL_LOC;
    otherGameProps.isHintPos = hintLoc != Board::NULL_LOC;
    otherGameProps.allowPolicyInit = hintLoc == Board::NULL_LOC; //On sgf positions, do allow extra moves at start
    otherGameProps.isFork = false;
    otherGameProps.isHintFork = false;
    otherGameProps.hintLoc = hintLoc;
    otherGameProps.hintTurn = (int)hist.moveHistory.size();
    otherGameProps.hintPosHash = hintPositionHash(board,hist);
    otherGameProps.trainingWeight = startPos.trainingWeight;
  }
  else {
    int xSize = allowedBSizes[bSizeIdx].first;
    int ySize = allowedBSizes[bSizeIdx].second;
    if(rules.isXiangqi()) {
      hist.clear(Xiangqi::State());
      board = hist.getRecentBoard(0);
      pla = hist.presumedNextMovePla;
    }
    else {
      board = Board(xSize,ySize);
      pla = P_BLACK;
      hist.clear(board,pla,rules,0);
    }

    numExtraBlack = rules.isGo() ? PlayUtils::chooseExtraBlack(handicapProb, numExtraBlackFixed, sqrt(board.x_size*board.y_size), rand) : 0;

    otherGameProps.isSgfPos = false;
    otherGameProps.isHintPos = false;
    otherGameProps.allowPolicyInit = true; //Handicap and regular games do allow policy init
    otherGameProps.isFork = false;
    otherGameProps.isHintFork = false;
    otherGameProps.hintLoc = Board::NULL_LOC;
    otherGameProps.hintTurn = -1;
    otherGameProps.trainingWeight = 1.0;
  }

  double asymmetricProb = (numExtraBlack > 0) ? playSettings.handicapAsymmetricPlayoutProb : playSettings.normalAsymmetricPlayoutProb;
  if(asymmetricProb > 0 && rand.nextBool(asymmetricProb)) {
    testAssert(playSettings.maxAsymmetricRatio >= 1.0);
    double maxNumDoublings = log(playSettings.maxAsymmetricRatio) / log(2.0);
    double numDoublings = rand.nextDouble(maxNumDoublings);
    if(numExtraBlack > 0 || rand.nextBool(0.5)) {
      otherGameProps.playoutDoublingAdvantagePla = C_WHITE;
      otherGameProps.playoutDoublingAdvantage = numDoublings;
    }
    else {
      otherGameProps.playoutDoublingAdvantagePla = C_BLACK;
      otherGameProps.playoutDoublingAdvantage = numDoublings;
    }
  }
}

//----------------------------------------------------------------------------------------------------------

MatchPairer::MatchPairer(
  ConfigParser& cfg,
  int nBots,
  const vector<string>& bNames,
  const vector<NNEvaluator*>& nEvals,
  const vector<SearchParams>& bParamss,
  const std::vector<std::pair<int,int>>& matchups,
  int64_t numGames
)
  :numBots(nBots),
   botNames(bNames),
   nnEvals(nEvals),
   baseParamss(bParamss),
   matchupsPerRound(matchups),
   nextMatchups(),
   rand(),
   numGamesStartedSoFar(0),
   numGamesTotal(numGames),
   logGamesEvery(),
   getMatchupMutex()
{
  testAssert(botNames.size() == numBots);
  testAssert(nnEvals.size() == numBots);
  testAssert(baseParamss.size() == numBots);

  if(matchupsPerRound.size() <= 0)
    throw StringError("MatchPairer: no matchups specified");
  if(matchupsPerRound.size() > 0xFFFFFF)
    throw StringError("MatchPairer: too many matchups");

  logGamesEvery = cfg.getInt64("logGamesEvery",1,1000000);
}

MatchPairer::~MatchPairer()
{}

int64_t MatchPairer::getNumGamesTotalToGenerate() const {
  return numGamesTotal;
}

bool MatchPairer::getMatchup(
  BotSpec& botSpecB, BotSpec& botSpecW, Logger& logger
)
{
  std::lock_guard<std::mutex> lock(getMatchupMutex);

  if(numGamesStartedSoFar >= numGamesTotal)
    return false;

  numGamesStartedSoFar += 1;

  if(numGamesStartedSoFar % logGamesEvery == 0)
    logger.write("Started " + Global::int64ToString(numGamesStartedSoFar) + " games");
  int64_t logNNEvery = logGamesEvery*100 > 1000 ? logGamesEvery*100 : 1000;
  if(numGamesStartedSoFar % logNNEvery == 0) {
    for(int i = 0; i<nnEvals.size(); i++) {
      if(nnEvals[i] != NULL) {
        logger.write(nnEvals[i]->getModelFileName());
        logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
        logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
        logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      }
    }
  }

  pair<int,int> matchup = getMatchupPairUnsynchronized();

  botSpecB.botIdx = matchup.first;
  botSpecB.botName = botNames[matchup.first];
  botSpecB.nnEval = nnEvals[matchup.first];
  botSpecB.baseParams = baseParamss[matchup.first];

  botSpecW.botIdx = matchup.second;
  botSpecW.botName = botNames[matchup.second];
  botSpecW.nnEval = nnEvals[matchup.second];
  botSpecW.baseParams = baseParamss[matchup.second];

  return true;
}

pair<int,int> MatchPairer::getMatchupPairUnsynchronized() {
  if(nextMatchups.size() <= 0) {
    if(numBots == 0)
      throw StringError("MatchPairer::getMatchupPairUnsynchronized: no bots to match up");

    //Append all matches for the next round
    nextMatchups.clear();
    nextMatchups.insert(nextMatchups.begin(), matchupsPerRound.begin(), matchupsPerRound.end());

    //Shuffle
    for(int i = (int)nextMatchups.size()-1; i >= 1; i--) {
      int j = (int)rand.nextUInt(i+1);
      pair<int,int> tmp = nextMatchups[i];
      nextMatchups[i] = nextMatchups[j];
      nextMatchups[j] = tmp;
    }
  }

  pair<int,int> matchup = nextMatchups.back();
  nextMatchups.pop_back();

  return matchup;
}

//----------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------

static void failIllegalMove(const Search* bot, Logger& logger, const Board& board, Loc loc) {
  ostringstream sout;
  sout << "Bot returned null location or illegal move!?!" << "\n";
  sout << board << "\n";
  sout << bot->getRootBoard() << "\n";
  sout << "Pla: " << PlayerIO::playerToString(bot->getRootPla()) << "\n";
  sout << "Loc: " << bot->moveToString(loc) << "\n";
  logger.write(sout.str());
  bot->getRootBoard().checkConsistency();
  ASSERT_UNREACHABLE;
}

static void logSearch(
  const Search* bot, Logger& logger, Loc loc, const OtherGameProperties& otherGameProps,
  const PrintTreeOptions& treeOptions
) {
  ostringstream sout;
  if(bot->getRootHist().rules.isXiangqi())
    bot->getRootHist().printDebugInfo(sout,bot->getRootBoard());
  else
    Board::printBoard(sout, bot->getRootBoard(), loc, &(bot->getRootHist().moveHistory));
  sout << "\n";
  sout << "Rules: " << bot->getRootHist().rules << "\n";
  sout << "Root visits: " << bot->getRootVisits() << "\n";
  if(otherGameProps.hintLoc != Board::NULL_LOC &&
     otherGameProps.hintTurn == bot->getRootHist().moveHistory.size() &&
     otherGameProps.hintPosHash == hintPositionHash(bot->getRootBoard(),bot->getRootHist())) {
    sout << "HintLoc " << bot->moveToString(otherGameProps.hintLoc) << "\n";
  }
  sout << "Policy surprise " << bot->getPolicySurprise() << "\n";
  sout << "Raw WL " << bot->getRootRawNNValuesRequireSuccess().winLossValue << "\n";
  sout << "PV: ";
  bot->printPV(sout, bot->rootNode, 25);
  sout << "\n";
  sout << "Tree:\n";
  bot->printTree(sout, bot->rootNode, treeOptions, P_WHITE);

  logger.write(sout.str());
}

static Loc chooseRandomForkingMove(const NNOutput* nnOutput, const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove) {
  double r = gameRand.nextDouble();
  bool allowPass = true;
  //70% of the time, do a random temperature 1 policy move
  if(r < 0.70)
    return PlayUtils::chooseRandomPolicyMove(nnOutput, board, hist, pla, gameRand, 1.0, allowPass, banMove);
  //25% of the time, do a random temperature 2 policy move
  else if(r < 0.95)
    return PlayUtils::chooseRandomPolicyMove(nnOutput, board, hist, pla, gameRand, 2.0, allowPass, banMove);
  //5% of the time, do a random legal move
  else
    return PlayUtils::chooseRandomLegalMove(board, hist, pla, gameRand, banMove);
}

bool Play::extractPolicyTarget(
  vector<PolicyTargetMove>& buf,
  const Search* toMoveBot,
  const SearchNode* node,
  vector<Loc>& locsBuf,
  vector<double>& playSelectionValuesBuf,
  const bool* policyMask
) {
  double scaleMaxToAtLeast = 10.0;

  testAssert(node != NULL);
  testAssert(!toMoveBot->searchParams.rootSymmetryPruning);
  bool allowDirectPolicyMoves = false;
  bool success = toMoveBot->searchParams.useGumbel ?
    toMoveBot->getPolicyTargetValues(*node,locsBuf,playSelectionValuesBuf) :
    toMoveBot->getPlaySelectionValues(*node,locsBuf,playSelectionValuesBuf,NULL,scaleMaxToAtLeast,allowDirectPolicyMoves);
  testAssert(success);

  testAssert(locsBuf.size() == playSelectionValuesBuf.size());
  testAssert(locsBuf.size() <= (size_t)(toMoveBot->getRootHist().rules.isXiangqi() ? Xiangqi::NUM_ACTIONS :
                                      toMoveBot->rootBoard.x_size * toMoveBot->rootBoard.y_size + 1));

  //Make sure we don't overflow int16
  double maxValue = 0.0;
  for(int moveIdx = 0; moveIdx<locsBuf.size(); moveIdx++) {
    if(policyMask && !policyMask[toMoveBot->getPos(locsBuf[moveIdx])]) {
      playSelectionValuesBuf[moveIdx] = 0.0;
      continue;
    }
    double value = playSelectionValuesBuf[moveIdx];
    testAssert(std::isfinite(value) && value >= 0.0);
    if(value > maxValue)
      maxValue = value;
  }

  double factor = 1.0;
  if(policyMask && maxValue <= 0.0) return false;
  if(policyMask && maxValue < scaleMaxToAtLeast) factor = scaleMaxToAtLeast / maxValue;
  if(maxValue > 30000.0 || toMoveBot->searchParams.useGumbel)
    factor = 30000.0 / maxValue;

  for(int moveIdx = 0; moveIdx<locsBuf.size(); moveIdx++) {
    if(policyMask && !policyMask[toMoveBot->getPos(locsBuf[moveIdx])]) continue;
    double value = playSelectionValuesBuf[moveIdx] * factor;
    assert(value <= 30001.0);
    buf.emplace_back(locsBuf[moveIdx],(int16_t)round(value));
  }
  return true;
}

static void extractValueTargets(ValueTargets& buf, const Search* toMoveBot, const SearchNode* node) {
  ReportedSearchValues values;
  bool success = toMoveBot->getNodeValues(node,values);
  testAssert(success);

  buf.win = (float)values.winValue;
  buf.loss = (float)values.lossValue;
  buf.draw = (float)values.drawValue;
  buf.noResult = (float)values.noResultValue;
}

static void extractQValueTargets(
  vector<QValueTargetMove>& buf,
  const Search* toMoveBot,
  const SearchNode* node
) {
  ConstSearchNodeChildrenReference children = node->getChildren();
  int numChildren = children.iterateAndCountChildren();
  for(int childIdx = 0; childIdx < numChildren; childIdx++) {
    const SearchChildPointer& childPtr = children[childIdx];
    const SearchNode* child = childPtr.getIfAllocated();

    if(child == NULL)
      continue;

    ReportedSearchValues values;
    bool success = toMoveBot->getNodeValues(child, values);
    if(!success)
      continue;
    if(values.visits <= 0)
      continue;

    Loc moveLoc = childPtr.getMoveLoc();
    buf.emplace_back(
      moveLoc,
      (float)values.winLossValue,
      values.visits
    );
  }
}

static NNRawStats computeNNRawStats(const Search* bot, const Board& board, const BoardHistory& hist, Player pla) {
  NNResultBuf buf;
  MiscNNInputParams nnInputParams;
  nnInputParams.drawEquivalentWinsForWhite = bot->searchParams.drawEquivalentWinsForWhite;
  //Featurize the way this bot's own searches would, even if the passed history differs.
  nnInputParams.passAliveSuicideRulesOverride =
    bot->mainEvaluatorHistoryModes().alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
  nnInputParams.excludeTerritoryAdjAtariOverride =
    bot->mainEvaluatorHistoryModes().excludeTerritoryAdjacentToAtari ? 1 : 0;
  bot->evaluatePosition(board,hist,pla,nnInputParams,buf,false);
  NNOutput& nnOutput = *(buf.result);

  NNRawStats nnRawStats;
  nnRawStats.whiteWinLoss = nnOutput.whiteWinProb - nnOutput.whiteLossProb;
  {
    double entropy = 0.0;
    int policySize = hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS : NNPos::getPolicySize(nnOutput.nnXLen,nnOutput.nnYLen);
    for(int pos = 0; pos<policySize; pos++) {
      double prob = nnOutput.policyProbs[pos];
      if(prob >= 1e-30)
        entropy += -prob * log(prob);
    }
    nnRawStats.policyEntropy = entropy;
  }
  return nnRawStats;
}

//One turn's worth of the per-turn training targets and stats recorded for selfplay training data, extracted
//from the search a bot just completed on a position. Extracted both for every turn during the game and when
//reanalyzing a position after the game - if adding a new per-turn target, make sure both get it.
struct SearchTargetsThisTurn {
  ValueTargets whiteValueTargets;
  QValueTargets whiteQValueTargets;
  std::vector<PolicyTargetMove>* policyTarget; //Receiver takes ownership
  int64_t unreducedNumVisits;
  NNRawStats nnRawStats;
  double policySurprise;
  double policyEntropy;
  double searchEntropy;
  ReportedSearchValues rawNNValues;
};

static void extractSearchTargetsThisTurn(
  SearchTargetsThisTurn& targets,
  Search* toMoveBot,
  const Board& board, const BoardHistory& hist, Player pla,
  vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf
) {
  extractValueTargets(targets.whiteValueTargets, toMoveBot, toMoveBot->rootNode);
  targets.whiteQValueTargets.targets.clear();
  extractQValueTargets(targets.whiteQValueTargets.targets, toMoveBot, toMoveBot->rootNode);
  targets.policyTarget = new vector<PolicyTargetMove>();
  Play::extractPolicyTarget(*targets.policyTarget, toMoveBot, toMoveBot->rootNode, locsBuf, playSelectionValuesBuf);
  targets.unreducedNumVisits = toMoveBot->getRootVisits();
  targets.nnRawStats = computeNNRawStats(toMoveBot, board, hist, pla);
  bool success = toMoveBot->getPolicySurpriseAndEntropy(targets.policySurprise, targets.searchEntropy, targets.policyEntropy);
  testAssert(success);
  (void)success; //Avoid warning when asserts are disabled
  targets.rawNNValues = toMoveBot->getRootRawNNValuesRequireSuccess();
}

//Recursively walk non-root-node subtree under node recording positions that have enough visits
//We also only record positions where the player to move made best moves along the tree so far.
//Does NOT walk down branches of excludeLoc0 and excludeLoc1 - these are used to avoid writing
//subtree positions for branches that we are about to actually play or do a forked sideposition search on.
static void recordTreePositionsRec(
  FinishedGameData* gameData,
  const Board& board, const BoardHistory& hist, Player pla,
  const Search* toMoveBot,
  const SearchNode* node, int depth, int maxDepth, bool plaAlwaysBest, bool oppAlwaysBest,
  int64_t minVisitsAtNode, float recordTreeTargetWeight,
  int numNeuralNetChangesSoFar,
  vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf,
  Loc excludeLoc0, Loc excludeLoc1
) {
  // Real-board replay is confined to sample extraction after the search. A
  // latent tree may continue past a real terminal or take an illegal action.
  if(toMoveBot->muZeroEvaluator && hist.isGameFinished) return;
  ConstSearchNodeChildrenReference children = node->getChildren();
  int numChildren = children.iterateAndCountChildren();

  if(numChildren <= 0)
    return;

  if(plaAlwaysBest && node != toMoveBot->rootNode) {
    SearchPolicyBuffer<bool> legalStorage(toMoveBot->muZeroEvaluator ? toMoveBot->getPolicyBufferSize() : 0);
    bool* legal = legalStorage.data();
    const bool* policyMask = nullptr;
    if(toMoveBot->muZeroEvaluator) {
      std::fill(legal,legal+toMoveBot->getPolicyBufferSize(),false);
      for(int pos = 0; pos < toMoveBot->policySize; pos++)
        legal[pos] = hist.isLegal(board,toMoveBot->getLoc(pos),pla);
      policyMask = legal;
    }
    auto sp = std::make_unique<SidePosition>(board,hist,pla,numNeuralNetChangesSoFar);
    bool hasPolicy = Play::extractPolicyTarget(sp->policyTarget, toMoveBot, node, locsBuf, playSelectionValuesBuf,policyMask);
    double policySurprise = 0.0, policyEntropy = 0.0, searchEntropy = 0.0;
    bool success = toMoveBot->getPolicySurpriseAndEntropy(policySurprise, searchEntropy, policyEntropy, node,policyMask);
    if(!policyMask) testAssert(success);
    // Without any real-legal target mass there is no root policy to distill.
    // Continue walking, since a deeper legally replayable node may have one.
    if(hasPolicy && success) {
      extractValueTargets(sp->whiteValueTargets, toMoveBot, node);
      extractQValueTargets(sp->whiteQValueTargets.targets, toMoveBot, node);
      if(policyMask) {
        auto& q = sp->whiteQValueTargets.targets;
        q.erase(std::remove_if(q.begin(),q.end(),[&](const QValueTargetMove& target) {
          return !policyMask[toMoveBot->getPos(target.loc)];
        }),q.end());
      }
      sp->policySurprise = policySurprise;
      sp->policyEntropy = policyEntropy;
      sp->searchEntropy = searchEntropy;
      sp->nnRawStats = computeNNRawStats(toMoveBot, board, hist, pla);
      sp->targetWeight = recordTreeTargetWeight;
      sp->unreducedNumVisits = toMoveBot->getRootVisits();
      gameData->sidePositions.push_back(sp.release());
    }
  }

  if(depth >= maxDepth)
    return;

  //Best child is the one with the largest number of visits, find it
  int bestChildIdx = 0;
  int64_t bestChildVisits = 0;
  for(int i = 1; i<numChildren; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    while(child->statsLock.test_and_set(std::memory_order_acquire));
    int64_t numVisits = child->stats.visits;
    child->statsLock.clear(std::memory_order_release);
    if(numVisits > bestChildVisits) {
      bestChildVisits = numVisits;
      bestChildIdx = i;
    }
  }

  for(int i = 0; i<numChildren; i++) {
    bool newPlaAlwaysBest = oppAlwaysBest;
    bool newOppAlwaysBest = plaAlwaysBest && i == bestChildIdx;

    if(!newPlaAlwaysBest && !newOppAlwaysBest)
      continue;

    const SearchNode* child = children[i].getIfAllocated();
    Loc moveLoc = children[i].getMoveLoc();
    if(moveLoc == excludeLoc0 || moveLoc == excludeLoc1)
      continue;

    int64_t numVisits = child->stats.visits.load(std::memory_order_acquire);
    if(numVisits < minVisitsAtNode)
      continue;

    if(hist.isLegal(board, moveLoc, pla)) {
      Board copy = board;
      BoardHistory histCopy = hist;
      histCopy.makeBoardMoveAssumeLegal(copy, moveLoc, pla, NULL);
      Player nextPla = getOpp(pla);
      recordTreePositionsRec(
        gameData,
        copy,histCopy,nextPla,
        toMoveBot,
        child,depth+1,maxDepth,newPlaAlwaysBest,newOppAlwaysBest,
        minVisitsAtNode,recordTreeTargetWeight,
        numNeuralNetChangesSoFar,
        locsBuf,playSelectionValuesBuf,
        Board::NULL_LOC,Board::NULL_LOC
      );
    }
  }
}

//Top-level caller for recursive func
static void recordTreePositions(
  FinishedGameData* gameData,
  const Board& board, const BoardHistory& hist, Player pla,
  const Search* toMoveBot,
  int64_t minVisitsAtNode, float recordTreeTargetWeight,
  int numNeuralNetChangesSoFar,
  vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf,
  Loc excludeLoc0, Loc excludeLoc1
) {
  testAssert(toMoveBot->rootBoard.pos_hash == board.pos_hash);
  testAssert(toMoveBot->rootHistory.moveHistory.size() == hist.moveHistory.size());
  testAssert(toMoveBot->rootPla == pla);
  testAssert(toMoveBot->rootNode != NULL);
  //Don't go too deep recording extra positions
  int maxDepth = 5;
  recordTreePositionsRec(
    gameData,
    board,hist,pla,
    toMoveBot,
    toMoveBot->rootNode, 0, maxDepth, true, true,
    minVisitsAtNode, recordTreeTargetWeight,
    numNeuralNetChangesSoFar,
    locsBuf,playSelectionValuesBuf,
    excludeLoc0,excludeLoc1
  );
}


struct SearchLimitsThisMove {
  bool doAlterVisitsPlayouts;
  int64_t numAlterVisits;
  int64_t numAlterPlayouts;
  bool clearBotBeforeSearchThisMove;
  bool removeRootNoise;
  float targetWeight;
  bool isCheapSearch;

  //Note: these two behave slightly differently than the ones in searchParams - derived from OtherGameProperties
  //game, they make the playouts *actually* vary instead of only making the neural net think they do.
  double playoutDoublingAdvantage;
  Player playoutDoublingAdvantagePla;

  Loc hintLoc;
};

//forceFullSearch skips the hint and random cheap search branches, giving the limits that a full-search turn
//would get, including any reduceVisits or asymmetric playout adjustments.
//Only the first numHistoricalMctsValuesToUse entries of historicalMctsWinLossValues count as the past for this
//turn - during the game that is all of them, while post-game reanalysis of a cheap-search position passes the
//turn index so as to reproduce the state as of that turn.
static SearchLimitsThisMove getSearchLimitsThisMove(
  const Search* toMoveBot, Player pla, const PlaySettings& playSettings, Rand& gameRand,
  const vector<double>& historicalMctsWinLossValues,
  size_t numHistoricalMctsValuesToUse,
  bool clearBotBeforeSearch,
  const OtherGameProperties& otherGameProps,
  bool forceFullSearch
) {
  bool doAlterVisitsPlayouts = false;
  int64_t numAlterVisits = toMoveBot->searchParams.maxVisits;
  int64_t numAlterPlayouts = toMoveBot->searchParams.maxPlayouts;
  bool clearBotBeforeSearchThisMove = clearBotBeforeSearch;
  bool removeRootNoise = false;
  float targetWeight = 1.0f;
  bool isCheapSearch = false;
  double playoutDoublingAdvantage = 0.0;
  Player playoutDoublingAdvantagePla = C_EMPTY;
  Loc hintLoc = Board::NULL_LOC;
  double cheapSearchProb = forceFullSearch ? 0.0 : playSettings.cheapSearchProb;

  const BoardHistory& hist = toMoveBot->getRootHist();
  if(!forceFullSearch && otherGameProps.hintLoc != Board::NULL_LOC) {
    if(otherGameProps.hintTurn == hist.moveHistory.size() &&
       otherGameProps.hintPosHash == hintPositionHash(toMoveBot->getRootBoard(),toMoveBot->getRootHist())) {
      hintLoc = otherGameProps.hintLoc;
      doAlterVisitsPlayouts = true;
      double cap = (double)((int64_t)1L << 50);
      numAlterVisits = (int64_t)ceil(std::min(cap, numAlterVisits * 4.0));
      numAlterPlayouts = (int64_t)ceil(std::min(cap, numAlterPlayouts * 4.0));
    }
  }
  //For the first few turns after a hint move or fork, reduce the probability of cheap search
  if((otherGameProps.hintLoc != Board::NULL_LOC || otherGameProps.isHintFork) && otherGameProps.hintTurn + 6 > hist.moveHistory.size()) {
    cheapSearchProb *= 0.5;
  }


  if(hintLoc == Board::NULL_LOC && cheapSearchProb > 0.0 && gameRand.nextBool(cheapSearchProb)) {
    if(playSettings.cheapSearchVisits <= 0)
      throw StringError("playSettings.cheapSearchVisits <= 0");
    if(playSettings.cheapSearchVisits > toMoveBot->searchParams.maxVisits ||
       playSettings.cheapSearchVisits > toMoveBot->searchParams.maxPlayouts)
      throw StringError("playSettings.cheapSearchVisits > maxVisits and/or maxPlayouts");

    doAlterVisitsPlayouts = true;
    isCheapSearch = true;
    numAlterVisits = std::min(numAlterVisits,(int64_t)playSettings.cheapSearchVisits);
    numAlterPlayouts = std::min(numAlterPlayouts,(int64_t)playSettings.cheapSearchVisits);
    targetWeight *= playSettings.cheapSearchTargetWeight;

    //If not recording cheap searches, do a few more things
    if(playSettings.cheapSearchTargetWeight <= 0.0) {
      clearBotBeforeSearchThisMove = false;
      removeRootNoise = true;
    }
  }
  else if(hintLoc == Board::NULL_LOC && playSettings.reduceVisits) {
    if(playSettings.reducedVisitsMin <= 0)
      throw StringError("playSettings.reducedVisitsMin <= 0");
    if(playSettings.reducedVisitsMin > toMoveBot->searchParams.maxVisits ||
       playSettings.reducedVisitsMin > toMoveBot->searchParams.maxPlayouts)
      throw StringError("playSettings.reducedVisitsMin > maxVisits and/or maxPlayouts");

    size_t numHistoricalValues = std::min(numHistoricalMctsValuesToUse, historicalMctsWinLossValues.size());
    if(numHistoricalValues >= (size_t)playSettings.reduceVisitsThresholdLookback) {
      double minWinLossValue = 1e20;
      double maxWinLossValue = -1e20;
      for(int j = 0; j<playSettings.reduceVisitsThresholdLookback; j++) {
        double winLossValue = historicalMctsWinLossValues[numHistoricalValues-1-j];
        if(winLossValue < minWinLossValue)
          minWinLossValue = winLossValue;
        if(winLossValue > maxWinLossValue)
          maxWinLossValue = winLossValue;
      }
      testAssert(playSettings.reduceVisitsThreshold >= 0.0);
      double signedMostExtreme = std::max(minWinLossValue,-maxWinLossValue);
      testAssert(signedMostExtreme <= 1.000001);
      if(signedMostExtreme > 1.0)
        signedMostExtreme = 1.0;
      double amountThrough = signedMostExtreme - playSettings.reduceVisitsThreshold;
      if(amountThrough > 0) {
        double proportionThrough = amountThrough / (1.0 - playSettings.reduceVisitsThreshold);
        testAssert(proportionThrough >= 0.0 && proportionThrough <= 1.0);
        double visitReductionProp = proportionThrough * proportionThrough;
        doAlterVisitsPlayouts = true;
        numAlterVisits = (int64_t)round(numAlterVisits + visitReductionProp * ((double)playSettings.reducedVisitsMin - (double)numAlterVisits));
        numAlterPlayouts = (int64_t)round(numAlterPlayouts + visitReductionProp * ((double)playSettings.reducedVisitsMin - (double)numAlterPlayouts));
        targetWeight = (float)(targetWeight + visitReductionProp * (playSettings.reducedVisitsWeight - targetWeight));
        numAlterVisits = std::max(numAlterVisits,(int64_t)playSettings.reducedVisitsMin);
        numAlterPlayouts = std::max(numAlterPlayouts,(int64_t)playSettings.reducedVisitsMin);
      }
    }
  }

  if(otherGameProps.playoutDoublingAdvantage != 0.0 && otherGameProps.playoutDoublingAdvantagePla != C_EMPTY) {
    testAssert(pla == otherGameProps.playoutDoublingAdvantagePla || getOpp(pla) == otherGameProps.playoutDoublingAdvantagePla);

    playoutDoublingAdvantage = otherGameProps.playoutDoublingAdvantage;
    playoutDoublingAdvantagePla = otherGameProps.playoutDoublingAdvantagePla;

    double factor = pow(2.0, otherGameProps.playoutDoublingAdvantage);
    if(pla == otherGameProps.playoutDoublingAdvantagePla)
      factor = 2.0 * (factor / (factor + 1.0));
    else
      factor = 2.0 * (1.0 / (factor + 1.0));


    doAlterVisitsPlayouts = true;
    //Set this back to true - we need to always clear the search if we are doing asymmetric playouts
    clearBotBeforeSearchThisMove = true;
    numAlterVisits = (int64_t)round(numAlterVisits * factor);
    numAlterPlayouts = (int64_t)round(numAlterPlayouts * factor);

    //Hardcoded limit here to ensure sanity
    if(numAlterVisits < 5)
      throw StringError("ERROR: asymmetric playout doubling resulted in fewer than 5 visits");
    if(numAlterPlayouts < 5)
      throw StringError("ERROR: asymmetric playout doubling resulted in fewer than 5 playouts");
  }

  SearchLimitsThisMove limits;
  limits.doAlterVisitsPlayouts = doAlterVisitsPlayouts;
  limits.numAlterVisits = numAlterVisits;
  limits.numAlterPlayouts = numAlterPlayouts;
  limits.clearBotBeforeSearchThisMove = clearBotBeforeSearchThisMove;
  limits.removeRootNoise = removeRootNoise;
  limits.targetWeight = targetWeight;
  limits.isCheapSearch = isCheapSearch;
  limits.playoutDoublingAdvantage = playoutDoublingAdvantage;
  limits.playoutDoublingAdvantagePla = playoutDoublingAdvantagePla;
  limits.hintLoc = hintLoc;
  return limits;
}

//Returns the move chosen
static Loc runBotWithLimits(
  Search* toMoveBot, Player pla, const PlaySettings& playSettings,
  const SearchLimitsThisMove& limits
) {
  if(limits.clearBotBeforeSearchThisMove)
    toMoveBot->clearSearch();

  Loc loc;

  //HACK - Disable LCB for making the move (it will still affect the policy target gen)
  bool lcb = toMoveBot->searchParams.useLcbForSelection;
  if(playSettings.forSelfPlay) {
    toMoveBot->searchParams.useLcbForSelection = false;
  }

  if(limits.doAlterVisitsPlayouts) {
    testAssert(limits.numAlterVisits > 0);
    testAssert(limits.numAlterPlayouts > 0);
    SearchParams oldParams = toMoveBot->searchParams;

    toMoveBot->searchParams.maxVisits = limits.numAlterVisits;
    toMoveBot->searchParams.maxPlayouts = limits.numAlterPlayouts;
    if(limits.removeRootNoise) {
      //Note - this is slightly sketchy to set the params directly. This works because
      //some of the parameters like FPU are basically stateless and will just affect future playouts
      //and because even stateful effects like rootNoiseEnabled and rootPolicyTemperature only affect
      //the root so when we step down in the tree we get a fresh start.
      toMoveBot->searchParams.rootNoiseEnabled = false;
      toMoveBot->searchParams.rootPolicyTemperature = 1.0;
      toMoveBot->searchParams.rootPolicyTemperatureEarly = 1.0;
      toMoveBot->searchParams.rootFpuLossProp = toMoveBot->searchParams.fpuLossProp;
      toMoveBot->searchParams.rootFpuReductionMax = toMoveBot->searchParams.fpuReductionMax;
      toMoveBot->searchParams.rootDesiredPerChildVisitsCoeff = 0.0;
      toMoveBot->searchParams.rootNumSymmetriesToSample = 1;
    }
    if(limits.playoutDoublingAdvantagePla != C_EMPTY) {
      toMoveBot->searchParams.playoutDoublingAdvantagePla = limits.playoutDoublingAdvantagePla;
      toMoveBot->searchParams.playoutDoublingAdvantage = limits.playoutDoublingAdvantage;
    }

    if(limits.hintLoc != Board::NULL_LOC) {
      testAssert(limits.clearBotBeforeSearchThisMove);
      //This will actually forcibly clear the search
      toMoveBot->setRootHintLoc(limits.hintLoc);
    }

    loc = toMoveBot->runWholeSearchAndGetMove(pla);

    if(limits.hintLoc != Board::NULL_LOC)
      toMoveBot->setRootHintLoc(Board::NULL_LOC);

    toMoveBot->searchParams = oldParams;
  }
  else {
    testAssert(!limits.removeRootNoise);
    loc = toMoveBot->runWholeSearchAndGetMove(pla);
  }

  //HACK - restore LCB so that it affects policy target gen
  if(playSettings.forSelfPlay) {
    toMoveBot->searchParams.useLcbForSelection = lcb;
  }

  return loc;
}

//KL divergence of a win/draw/loss/noresult distribution from the raw neural net value prediction, capped at 1.
static double valueSurpriseKL(double winValue, double drawValue, double lossValue, double noResultValue, const ReportedSearchValues& rawNNValues) {
  double valueSurprise = 0.0;
  if(winValue > 1e-100) valueSurprise += winValue * (log(winValue) - log(std::max((double)rawNNValues.winValue,1e-100)));
  if(lossValue > 1e-100) valueSurprise += lossValue * (log(lossValue) - log(std::max((double)rawNNValues.lossValue,1e-100)));
  if(drawValue > 1e-100) valueSurprise += drawValue * (log(drawValue) - log(std::max((double)rawNNValues.drawValue,1e-100)));
  if(noResultValue > 1e-100) valueSurprise += noResultValue * (log(noResultValue) - log(std::max((double)rawNNValues.noResultValue,1e-100)));

  //Just in case, guard against float imprecision
  if(valueSurprise < 0.0)
    valueSurprise = 0.0;
  //Cap value surprise at extreme value, to reduce the chance of a ridiculous weight on a move.
  return std::min(valueSurprise,1.0);
}

//Compute how surprising the value result at each turn was (KL divergence), relative to the raw neural net
//value prediction at that turn. By default the value result compared is the smoothed forward-looking game
//result, which depends on how the game actually continued and ended. If useSearchValueSurprise is true, it is
//instead the value result of that turn's own search, which depends only on information available at the time
//of the search, so that data weighting/selection based on it does not condition on the game's realized outcome.
//whiteValueTargetsByTurn should have one entry per turn plus a final entry for the game outcome.
static void computeValueSurpriseByTurn(
  vector<double>& valueSurpriseByTurn,
  const vector<ValueTargets>& whiteValueTargetsByTurn,
  const vector<ReportedSearchValues>& rawNNValues,
  int boardArea,
  bool useSearchValueSurprise
) {
  testAssert(whiteValueTargetsByTurn.size() == rawNNValues.size() + 1);
  valueSurpriseByTurn.resize(rawNNValues.size());

  if(useSearchValueSurprise) {
    for(int i = 0; i < (int)rawNNValues.size(); i++) {
      const ValueTargets& searchValues = whiteValueTargetsByTurn[i];
      valueSurpriseByTurn[i] = valueSurpriseKL(searchValues.win, searchValues.draw, searchValues.loss, searchValues.noResult, rawNNValues[i]);
    }
    return;
  }

  double nowFactor = 1.0/(1.0 + boardArea * 0.016);

  double winValue = whiteValueTargetsByTurn[whiteValueTargetsByTurn.size()-1].win;
  double lossValue = whiteValueTargetsByTurn[whiteValueTargetsByTurn.size()-1].loss;
  double drawValue = whiteValueTargetsByTurn[whiteValueTargetsByTurn.size()-1].draw;
  double noResultValue = whiteValueTargetsByTurn[whiteValueTargetsByTurn.size()-1].noResult;
  for(int i = (int)rawNNValues.size()-1; i >= 0; i--) {
    winValue = winValue + nowFactor * (whiteValueTargetsByTurn[i].win - winValue);
    lossValue = lossValue + nowFactor * (whiteValueTargetsByTurn[i].loss - lossValue);
    drawValue = drawValue + nowFactor * (whiteValueTargetsByTurn[i].draw - drawValue);
    noResultValue = noResultValue + nowFactor * (whiteValueTargetsByTurn[i].noResult - noResultValue);

    valueSurpriseByTurn[i] = valueSurpriseKL(winValue, drawValue, lossValue, noResultValue, rawNNValues[i]);
  }
}

//After a game finishes, select some of the positions that got only cheap searches, favoring positions whose
//cheap search was surprising, and redo them with full searches. Overwrites those turns' recorded targets and
//stats with those of the new full searches and raises their target weights to what a full-search turn would get,
//so that they get recorded as full-quality training positions. Does not affect the game itself in any way.
static void reanalyzeCheapSearchPositions(
  Search* botB, Search* botW,
  const PlaySettings& playSettings,
  const OtherGameProperties& otherGameProps,
  bool clearBotBeforeSearch,
  const vector<bool>& wasCheapSearchByTurn,
  const vector<double>& valueSurpriseByTurn,
  const vector<double>& historicalMctsWinLossValues,
  Rand& gameRand,
  const std::function<bool()>& shouldStop,
  const WaitableFlag* shouldPause,
  const std::function<void(int)>& maybeCheckForNewNNEval,
  vector<ReportedSearchValues>& rawNNValues,
  vector<Loc>& locsBuf, vector<double>& playSelectionValuesBuf,
  FinishedGameData* gameData
) {
  int numMoves = (int)gameData->targetWeightByTurn.size();
  testAssert((int)wasCheapSearchByTurn.size() == numMoves);
  testAssert((int)valueSurpriseByTurn.size() == numMoves);
  testAssert((int)gameData->policySurpriseByTurn.size() == numMoves);
  testAssert((int)gameData->reanalysisByTurn.size() == numMoves);

  vector<int> candidateTurns;
  for(int i = 0; i<numMoves; i++)
    if(wasCheapSearchByTurn[i])
      candidateTurns.push_back(i);
  if(candidateTurns.size() <= 0)
    return;

  //Binomial draw on the number of cheap search positions determines how many to reanalyze.
  int numToReanalyze = 0;
  for(size_t i = 0; i<candidateTurns.size(); i++)
    if(gameRand.nextBool(playSettings.reanalyzeProp))
      numToReanalyze += 1;
  if(numToReanalyze <= 0)
    return;

  //Sample that many turns without replacement, proportionally to how surprising the cheap search was.
  vector<double> relProbs(candidateTurns.size());
  for(size_t i = 0; i<candidateTurns.size(); i++) {
    int turn = candidateTurns[i];
    double surprise =
      playSettings.reanalyzePolicySurpriseWeight * gameData->policySurpriseByTurn[turn]
      + playSettings.reanalyzeValueSurpriseWeight * valueSurpriseByTurn[turn];
    relProbs[i] = pow(surprise, playSettings.reanalyzeSurpriseExponent);
  }
  vector<int> turnsToReanalyze;
  for(int k = 0; k<numToReanalyze; k++) {
    double probSum = 0.0;
    for(size_t i = 0; i<relProbs.size(); i++)
      probSum += relProbs[i];
    size_t chosen;
    if(probSum > 1e-30)
      chosen = gameRand.nextUInt(relProbs.data(),relProbs.size());
    else
      chosen = gameRand.nextUInt((uint32_t)relProbs.size());
    turnsToReanalyze.push_back(candidateTurns[chosen]);
    candidateTurns.erase(candidateTurns.begin()+chosen);
    relProbs.erase(relProbs.begin()+chosen);
  }
  std::sort(turnsToReanalyze.begin(),turnsToReanalyze.end());

  //Replay the game forward, redoing a full search at each selected turn.
  Board board = gameData->startBoard;
  BoardHistory hist = gameData->startHist;
  Player pla = gameData->startPla;
  int startTurnIdx = (int)gameData->startHist.moveHistory.size();

  size_t whichIdx = 0;
  for(int turnAfterStart = 0; turnAfterStart<numMoves && whichIdx < turnsToReanalyze.size(); turnAfterStart++) {
    if(turnAfterStart == turnsToReanalyze[whichIdx]) {
      whichIdx += 1;
      if(shouldPause != nullptr)
        shouldPause->waitUntilFalse();
      if(shouldStop != nullptr && shouldStop())
        break;

      Search* toMoveBot = pla == P_BLACK ? botB : botW;
      toMoveBot->setPosition(pla,board,hist);

      //Reproduce the search limits that this turn would have gotten if it had been a full-search turn,
      //including any reduceVisits and asymmetric playout adjustments, based on the mcts winloss values
      //of the turns leading up to this one.
      SearchLimitsThisMove limits = getSearchLimitsThisMove(
        toMoveBot, pla, playSettings, gameRand, historicalMctsWinLossValues, (size_t)turnAfterStart,
        clearBotBeforeSearch, otherGameProps, true
      );
      runBotWithLimits(toMoveBot, pla, playSettings, limits);

      //Record the original cheap search's surprise stats that drove the selection of this position, and its visits.
      ReanalysisData& reanalysisData = gameData->reanalysisByTurn[turnAfterStart];
      reanalysisData.wasReanalyzed = true;
      reanalysisData.usedOutcomeTargets = playSettings.reanalyzeUseOutcomeTargets;
      reanalysisData.selectionPolicySurprise = (float)gameData->policySurpriseByTurn[turnAfterStart];
      reanalysisData.selectionValueSurprise = (float)valueSurpriseByTurn[turnAfterStart];
      reanalysisData.originalNumVisits = gameData->policyTargetsByTurn[turnAfterStart].unreducedNumVisits;
      reanalysisData.numNeuralNetChangesSoFar = (int)gameData->changedNeuralNets.size();

      //Overwrite this turn's recorded targets and stats with those of the new full search.
      //Note: replacing this turn's entry in whiteValueTargetsByTurn means the winloss-variance-timing training
      //target of earlier turns will see cheap-vs-full estimator deltas at the boundaries of this turn that
      //aren't purely game variance. Deltas of similar scale already pervade that target from consecutive
      //cheap/full searches during play, and it is a tiny auxiliary target, so this is accepted.
      SearchTargetsThisTurn targets;
      extractSearchTargetsThisTurn(targets, toMoveBot, board, hist, pla, locsBuf, playSelectionValuesBuf);
      gameData->whiteValueTargetsByTurn[turnAfterStart] = targets.whiteValueTargets;
      gameData->whiteQValueTargetsByTurn[turnAfterStart] = targets.whiteQValueTargets;
      delete gameData->policyTargetsByTurn[turnAfterStart].policyTargets;
      gameData->policyTargetsByTurn[turnAfterStart].policyTargets = targets.policyTarget;
      gameData->policyTargetsByTurn[turnAfterStart].unreducedNumVisits = targets.unreducedNumVisits;
      gameData->nnRawStatsByTurn[turnAfterStart] = targets.nnRawStats;
      gameData->policySurpriseByTurn[turnAfterStart] = targets.policySurprise;
      gameData->policyEntropyByTurn[turnAfterStart] = targets.policyEntropy;
      gameData->searchEntropyByTurn[turnAfterStart] = targets.searchEntropy;
      rawNNValues[turnAfterStart] = targets.rawNNValues;
      gameData->targetWeightByTurn[turnAfterStart] = limits.targetWeight;

      testAssert(gameData->endHist.moveHistory.size() < 0x1FFFffff);
      maybeCheckForNewNNEval((int)gameData->endHist.moveHistory.size());
    }

    Move move = gameData->endHist.moveHistory[turnAfterStart + startTurnIdx];
    testAssert(move.pla == pla);
    testAssert(hist.isLegal(board,move.loc,move.pla));
    hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, NULL);
    pla = getOpp(pla);
  }
}


Search* MatchPairer::BotSpec::createSearch(Logger& logger, const string& seed) const {
  if(muZeroEval) {
    if(nnEval) throw StringError("BotSpec cannot contain both board and MuZero evaluators");
    return new Search(baseParams,*muZeroEval,&logger,seed);
  }
  return new Search(baseParams,nnEval,&logger,seed);
}

BoardHistoryModes MatchPairer::BotSpec::historyModes() const {
  return muZeroEval ? Search::resolveHistoryModes(baseParams,*muZeroEval) : Search::resolveHistoryModes(baseParams,nnEval);
}

//Run a game between two bots. It is OK if both bots are the same bot.
FinishedGameData* Play::runGame(
  const Board& startBoard, Player pla, const BoardHistory& startHist, int numExtraBlack,
  MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
  const string& searchRandSeed,
  bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, const std::function<bool()>& shouldStop,
  const WaitableFlag* shouldPause,
  const PlaySettings& playSettings, const OtherGameProperties& otherGameProps,
  Rand& gameRand,
  const std::function<NNEvaluator*()>& checkForNewNNEval,
  const std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const Search*)>& onEachMove
) {
  Search* botB;
  Search* botW;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    botB = botSpecB.createSearch(logger,searchRandSeed);
    botW = botB;
  }
  else {
    botB = botSpecB.createSearch(logger,searchRandSeed + "@B");
    botW = botSpecW.createSearch(logger,searchRandSeed + "@W");
  }

  FinishedGameData* gameData = runGame(
    startBoard, pla, startHist, numExtraBlack,
    botSpecB, botSpecW,
    botB, botW,
    doEndGameIfAllPassAlive, clearBotBeforeSearch,
    logger, logSearchInfo, logMoves,
    maxMovesPerGame, shouldStop,
    shouldPause,
    playSettings, otherGameProps,
    gameRand,
    checkForNewNNEval,
    onEachMove
  );

  if(botW != botB)
    delete botW;
  delete botB;

  return gameData;
}

FinishedGameData* Play::runGame(
  const Board& startBoard, Player startPla, const BoardHistory& startHist, int numExtraBlack,
  MatchPairer::BotSpec& botSpecB, MatchPairer::BotSpec& botSpecW,
  Search* botB, Search* botW,
  bool doEndGameIfAllPassAlive, bool clearBotBeforeSearch,
  Logger& logger, bool logSearchInfo, bool logMoves,
  int maxMovesPerGame, const std::function<bool()>& shouldStop,
  const WaitableFlag* shouldPause,
  const PlaySettings& playSettings, const OtherGameProperties& otherGameProps,
  Rand& gameRand,
  const std::function<NNEvaluator*()>& checkForNewNNEval,
  const std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const Search*)>& onEachMove
) {
  if(botB->muZeroEvaluator || botW->muZeroEvaluator) {
    if(checkForNewNNEval)
      throw StringError("MuZero selfplay uses a fixed model for the game; mid-game model switching is not connected");
  }
  FinishedGameData* gameData = new FinishedGameData();

  Board board(startBoard);
  BoardHistory hist(startHist);
  //Game-level BoardHistoryModes, used for whole-game adjudication (endGameIfAllPassAlive,
  //final scoring) and training data targets: use each new-behavior mode only if BOTH
  //bots' searches will be using it. Each bot's Search still stamps its own resolved settings onto its
  //internal copy of the history, which may differ from this when the two bots' settings differ.
  {
    const BoardHistoryModes modesB = botSpecB.historyModes();
    const BoardHistoryModes modesW = botSpecW.historyModes();
    hist.setModes(BoardHistoryModes(
      modesB.alwaysComputePassAliveUnderSuicideRules && modesW.alwaysComputePassAliveUnderSuicideRules,
      modesB.excludeTerritoryAdjacentToAtari && modesW.excludeTerritoryAdjacentToAtari
    ));
  }
  Player pla = startPla;
  testAssert(!(playSettings.forSelfPlay && !clearBotBeforeSearch));

  if(numExtraBlack > 0 && !hist.isGameFinished) {
    double extraBlackTemperature = playSettings.handicapTemperature;
    testAssert(extraBlackTemperature > 0.0 && extraBlackTemperature < 10.0);
    PlayUtils::playExtraBlack(botB,numExtraBlack,board,hist,extraBlackTemperature,gameRand);
    testAssert(hist.moveHistory.size() == 0);
  }
  gameData->bName = botSpecB.botName;
  gameData->wName = botSpecW.botName;
  gameData->bIdx = botSpecB.botIdx;
  gameData->wIdx = botSpecW.botIdx;

  gameData->gameHash.hash0 = gameRand.nextUInt64();
  gameData->gameHash.hash1 = gameRand.nextUInt64();

  gameData->drawEquivalentWinsForWhite = botSpecB.baseParams.drawEquivalentWinsForWhite;
  gameData->playoutDoublingAdvantagePla = otherGameProps.playoutDoublingAdvantagePla;
  gameData->playoutDoublingAdvantage = otherGameProps.playoutDoublingAdvantage;

  gameData->numExtraBlack = numExtraBlack;
  gameData->handicapForSgf = numExtraBlack; //overwritten later
  gameData->mode = FinishedGameData::MODE_NORMAL;
  gameData->beganInEncorePhase = 0;
  gameData->usedInitialPosition = 0;

  if(numExtraBlack > 0)
    gameData->mode = FinishedGameData::MODE_HANDICAP;

  //Might get overwritten next as we also play sgfposes and such with asym mode!
  //So this is just a best efforts to make it more prominent for most of the asymmetric games.
  if(gameData->playoutDoublingAdvantage != 0)
    gameData->mode = FinishedGameData::MODE_ASYM;

  if(otherGameProps.isSgfPos)
    gameData->mode = FinishedGameData::MODE_SGFPOS;
  if(otherGameProps.isHintPos)
    gameData->mode = FinishedGameData::MODE_HINTPOS;

  if(otherGameProps.isHintFork)
    gameData->mode = FinishedGameData::MODE_HINTFORK;
  else if(otherGameProps.isFork)
    gameData->mode = FinishedGameData::MODE_FORK;

  //In selfplay, record all the policy maps and evals and such as well for training data
  bool recordFullData = playSettings.forSelfPlay;

  //NOTE: that checkForNewNNEval might also cause the old nnEval to be invalidated and freed. This is okay since the only
  //references we both hold on to and use are the ones inside the bots here, and we replace the ones in the botSpecs.
  //We should NOT ever store an nnEval separately from these.
  auto maybeCheckForNewNNEval = [&botB,&botW,&botSpecB,&botSpecW,&checkForNewNNEval,&gameRand,&gameData](int nextTurnIdx) {
    //Check if we got a new nnEval, with some probability.
    //Randomized and low-probability so as to reduce contention in checking, while still probably happening in a timely manner.
    //NOTE: swapping the nnEval mid-game re-resolves each bot's alwaysComputePassAliveUnderSuicideRules mode
    //(via Search::setNNEval), but the game-level hist deliberately KEEPS the mode stamped at the start of the
    //game - re-adjudicating a game under a different mode than it started with would be worse than letting
    //the searches briefly disagree with the game-level adjudication for the remainder of this one game.
    if(checkForNewNNEval != nullptr && gameRand.nextBool(0.1)) {
      NNEvaluator* newNNEval = checkForNewNNEval();
      if(newNNEval != NULL) {
        botB->setNNEval(newNNEval);
        if(botW != botB)
          botW->setNNEval(newNNEval);
        botSpecB.nnEval = newNNEval;
        botSpecW.nnEval = newNNEval;
        gameData->changedNeuralNets.emplace_back(new ChangedNeuralNet(newNNEval->getModelName(),nextTurnIdx));
      }
    }
  };

  const bool gomokuOpening = hist.rules.isPlacement() && hist.rules.game != Rules::Game::HEX;
  const bool hexSetup = hist.rules.game == Rules::Game::HEX &&
    (playSettings.balancedOpening.specialOpeningProb > 0.0 || playSettings.balancedOpening.completelyRandomOpeningProb > 0.0);
  if((playSettings.balancedOpening.probability > 0.0 || hexSetup) &&
     (gomokuOpening || (!otherGameProps.isSgfPos && !otherGameProps.isHintPos &&
                       !otherGameProps.isFork && !otherGameProps.isHintFork))) {
    const auto result = BalancedOpening::initialize(botB,botW,board,hist,pla,gameRand,playSettings.balancedOpening,shouldStop);
    if(result == BalancedOpening::Result::INTERRUPTED) {
      // GameRunner discards interrupted games before writing data or forking.
      gameData->startBoard = board;
      gameData->startHist = hist;
      gameData->endHist = hist;
      gameData->startPla = pla;
      gameData->trainingWeight = otherGameProps.trainingWeight;
      return gameData;
    }
    if(result == BalancedOpening::Result::APPLIED)
      logger.write("Balanced opening: " + Rules::gameName(hist.rules.game) + " moves=" + Global::intToString(hist.moveHistory.size()));
    else if(result == BalancedOpening::Result::APPLIED_SPECIAL || result == BalancedOpening::Result::APPLIED_RANDOM)
      logger.write(std::string("Hex opening: ") + (result == BalancedOpening::Result::APPLIED_SPECIAL ? "special" : "random-fill") +
                   " setupStones=" + Global::intToString(board.numStonesOnBoard()));
  }

  if(playSettings.initGamesWithPolicy && otherGameProps.allowPolicyInit && !hist.isGameFinished) {
    const double mean = hist.rules.isPlacement() ?
      (otherGameProps.isSgfPos ? playSettings.startPosesPolicyInitAvgMoveNum : playSettings.policyInitAvgMoveNum) :
      board.x_size * board.y_size * (otherGameProps.isSgfPos ? playSettings.startPosesPolicyInitAreaProp : playSettings.policyInitAreaProp);
    if(mean > 0) {
      double policyInitGammaShape = playSettings.policyInitGammaShape;
      double temperature = playSettings.policyInitAreaTemperature;
      testAssert(temperature > 0.0 && temperature < 10.0);
      PlayUtils::initializeGameUsingPolicy(botB, botW, board, hist, pla, gameRand, doEndGameIfAllPassAlive, mean, policyInitGammaShape, temperature);
    }
  }

  //Make sure there's some minimum tiny amount of data about how the encore phases work
  if(
    playSettings.forSelfPlay &&
    !otherGameProps.isHintPos &&
    hist.rules.scoringRule == Rules::SCORING_TERRITORY &&
    hist.encorePhase == 0 &&
    gameRand.nextBool(0.04) &&
    !hist.isGameFinished
  ) {
    //Play out to go a quite a bit later in the game.
    double proportionOfBoardArea = 0.25;
    double policyInitGammaShape = 1.0 * 0.8 + playSettings.policyInitGammaShape * 0.2;
    double temperature = 2.0/3.0;
    PlayUtils::initializeGameUsingPolicy(botB, botW, board, hist, pla, gameRand, doEndGameIfAllPassAlive, board.x_size * board.y_size * proportionOfBoardArea, policyInitGammaShape, temperature);

    if(!hist.isGameFinished) {
      //Randomly set to one of the encore phases
      //Since we played out the game a bunch we should get a good mix of stones that were present or not present at the start
      //of the second encore phase if we're going into the second.
      int encorePhase = gameRand.nextInt(1,2);
      board.clearSimpleKoLoc();
      hist.clear(board,pla,hist.rules,encorePhase);

      gameData->mode = FinishedGameData::MODE_CLEANUP_TRAINING;
      gameData->beganInEncorePhase = encorePhase;
      gameData->usedInitialPosition = 0;
    }
  }

  //Set in the starting board and history to gameData and both bots
  gameData->startBoard = board;
  gameData->startHist = hist;
  gameData->startPla = pla;

  botB->setPosition(pla,board,hist);
  if(botB != botW)
    botW->setPosition(pla,board,hist);

  vector<Loc> locsBuf;
  vector<double> playSelectionValuesBuf;

  vector<SidePosition*> sidePositionsToSearch;

  vector<double> historicalMctsWinLossValues;
  vector<ReportedSearchValues> rawNNValues;
  vector<bool> wasCheapSearchByTurn;

  ClockTimer timer;

  //Main play loop
  for(int i = 0; i<maxMovesPerGame; i++) {
    if(doEndGameIfAllPassAlive)
      hist.endGameIfAllPassAlive(board);
    if(hist.isGameFinished)
      break;
    if(shouldPause != nullptr)
      shouldPause->waitUntilFalse();
    if(shouldStop != nullptr && shouldStop())
      break;

    Search* toMoveBot = pla == P_BLACK ? botB : botW;
    SearchLimitsThisMove limits = getSearchLimitsThisMove(
      toMoveBot, pla, playSettings, gameRand, historicalMctsWinLossValues, historicalMctsWinLossValues.size(),
      clearBotBeforeSearch, otherGameProps, false
    );
    Loc loc;
    if(playSettings.recordTimePerMove) {
      double t0 = timer.getSeconds();
      loc = runBotWithLimits(toMoveBot, pla, playSettings, limits);
      double t1 = timer.getSeconds();
      if(pla == P_BLACK)
        gameData->bTimeUsed += t1-t0;
      else
        gameData->wTimeUsed += t1-t0;
    }
    else {
      loc = runBotWithLimits(toMoveBot, pla, playSettings, limits);
    }

    if(pla == P_BLACK)
      gameData->bMoveCount += 1;
    else
      gameData->wMoveCount += 1;

    if(loc == Board::NULL_LOC || !toMoveBot->isLegalStrict(loc,pla))
      failIllegalMove(toMoveBot,logger,board,loc);
    if(logSearchInfo)
      logSearch(toMoveBot,logger,loc,otherGameProps,playSettings.logSearchTreeOptions);
    if(logMoves)
      logger.write("Move " + Global::uint64ToString(hist.moveHistory.size()) + " made: " + toMoveBot->moveToString(loc));

    if(!recordFullData) {
      ValueTargets whiteValueTargets;
      extractValueTargets(whiteValueTargets, toMoveBot, toMoveBot->rootNode);
      gameData->whiteValueTargetsByTurn.push_back(whiteValueTargets);
      QValueTargets whiteQValueTargets;
      extractQValueTargets(whiteQValueTargets.targets, toMoveBot, toMoveBot->rootNode);
      gameData->whiteQValueTargetsByTurn.push_back(whiteQValueTargets);
      //Go ahead and record this anyways with just the visits, as a bit of a hack so that the sgf output can also write the number of visits.
      int64_t unreducedNumVisits = toMoveBot->getRootVisits();
      gameData->policyTargetsByTurn.emplace_back(nullptr,unreducedNumVisits);
    }
    else {
      SearchTargetsThisTurn targets;
      extractSearchTargetsThisTurn(targets, toMoveBot, board, hist, pla, locsBuf, playSelectionValuesBuf);
      gameData->whiteValueTargetsByTurn.push_back(targets.whiteValueTargets);
      gameData->whiteQValueTargetsByTurn.push_back(targets.whiteQValueTargets);
      gameData->policyTargetsByTurn.emplace_back(targets.policyTarget,targets.unreducedNumVisits);
      gameData->nnRawStatsByTurn.push_back(targets.nnRawStats);
      gameData->policySurpriseByTurn.push_back(targets.policySurprise);
      gameData->policyEntropyByTurn.push_back(targets.policyEntropy);
      gameData->searchEntropyByTurn.push_back(targets.searchEntropy);
      rawNNValues.push_back(targets.rawNNValues);

      gameData->targetWeightByTurn.push_back(limits.targetWeight);
      gameData->reanalysisByTurn.emplace_back();
      wasCheapSearchByTurn.push_back(limits.isCheapSearch);

      //Occasionally fork off some positions to evaluate
      Loc sidePositionForkLoc = Board::NULL_LOC;
      if(playSettings.sidePositionProb > 0.0 && gameRand.nextBool(playSettings.sidePositionProb)) {
        testAssert(toMoveBot->rootNode != NULL);
        const NNOutput* nnOutput = toMoveBot->rootNode->getNNOutput();
        testAssert(nnOutput != NULL);
        Loc banMove = loc;
        sidePositionForkLoc = chooseRandomForkingMove(nnOutput, board, hist, pla, gameRand, banMove);
        if(sidePositionForkLoc != Board::NULL_LOC) {
          SidePosition* sp = new SidePosition(board,hist,pla,(int)gameData->changedNeuralNets.size());
          sp->hist.makeBoardMoveAssumeLegal(sp->board,sidePositionForkLoc,sp->pla,NULL);
          sp->pla = getOpp(sp->pla);
          if(sp->hist.isGameFinished) delete sp;
          else sidePositionsToSearch.push_back(sp);
        }
      }

      //If enabled, also record subtree positions from the search as training positions
      if(playSettings.recordTreePositions && playSettings.recordTreeTargetWeight > 0.0f) {
        if(playSettings.recordTreeTargetWeight > 1.0f)
          throw StringError("playSettings.recordTreeTargetWeight > 1.0f");

        recordTreePositions(
          gameData,
          board,hist,pla,
          toMoveBot,
          playSettings.recordTreeThreshold,playSettings.recordTreeTargetWeight,
          (int)gameData->changedNeuralNets.size(),
          locsBuf,playSelectionValuesBuf,
          loc,sidePositionForkLoc
        );
      }
    }

    if(playSettings.allowResignation || playSettings.reduceVisits) {
      ReportedSearchValues values = toMoveBot->getRootValuesRequireSuccess();
      historicalMctsWinLossValues.push_back(values.winLossValue);
    }

    if(onEachMove != nullptr)
      onEachMove(board,hist,pla,loc,historicalMctsWinLossValues,toMoveBot);

    //Finally, make the move on the bots
    bool suc;
    suc = botB->makeMove(loc,pla);
    testAssert(suc);
    if(botB != botW) {
      suc = botW->makeMove(loc,pla);
      testAssert(suc);
    }

    //And make the move on our copy of the board
    testAssert(hist.isLegal(board,loc,pla));
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);

    //Check for resignation
    if(playSettings.allowResignation && historicalMctsWinLossValues.size() >= playSettings.resignConsecTurns) {
      //Play at least some moves no matter what
      int minTurnForResignation = 1 + board.x_size * board.y_size / 5;
      if(i >= minTurnForResignation) {
        if(playSettings.resignThreshold > 0 || std::isnan(playSettings.resignThreshold))
          throw StringError("playSettings.resignThreshold > 0 || std::isnan(playSettings.resignThreshold)");

        bool shouldResign = true;
        for(int j = 0; j<playSettings.resignConsecTurns; j++) {
          double winLossValue = historicalMctsWinLossValues[historicalMctsWinLossValues.size()-j-1];
          Player resignPlayerThisTurn = C_EMPTY;
          if(winLossValue < playSettings.resignThreshold)
            resignPlayerThisTurn = P_WHITE;
          else if(winLossValue > -playSettings.resignThreshold)
            resignPlayerThisTurn = P_BLACK;

          if(resignPlayerThisTurn != pla) {
            shouldResign = false;
            break;
          }
        }

        if(shouldResign)
          hist.setWinnerByResignation(getOpp(pla));
      }
    }

    testAssert(hist.moveHistory.size() < 0x1FFFffff);
    int nextTurnIdx = (int)hist.moveHistory.size();
    maybeCheckForNewNNEval(nextTurnIdx);

    pla = getOpp(pla);
  }

  gameData->endHist = hist;
  if(hist.isGameFinished)
    gameData->hitTurnLimit = false;
  else
    gameData->hitTurnLimit = true;

  // In self-play or match play, it should ALWAYS be the case that the entire game history is legal.
  if(hist.numConsecValidTurnsThisGame != hist.moveHistory.size()) {
    ostringstream sout;
    sout << "Selfplay got history with not entire game legal!?!" << "\n";
    sout << "hist.numConsecValidTurnsThisGame " << hist.numConsecValidTurnsThisGame << "\n";
    sout << "hist.moveHistory.size() " << hist.moveHistory.size() << "\n";
    hist.printBasicInfo(sout,board);
    hist.printDebugInfo(sout,board);
    logger.write(sout.str());
    cerr << sout.str() << endl;
    testAssert(false);
  }

  {
    BoardHistory histCopy(hist);
    //Always use true for computing the handicap value that goes into an sgf
    histCopy.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    gameData->handicapForSgf = histCopy.computeNumHandicapStones();
  }

  if(recordFullData) {
    if(hist.isResignation)
      throw StringError("Recording full data currently incompatible with resignation");

    ValueTargets finalValueTargets;


    if(hist.isGameFinished && hist.isNoResult) {
      finalValueTargets.win = 0.0f;
      finalValueTargets.loss = 0.0f;
      finalValueTargets.noResult = 1.0f;

    }
    else if(!hist.rules.isGo() && !hist.isGameFinished) {
      // A move limit does not adjudicate placement games. Bootstrap TD from the last real state.
      Search* bot = pla == P_BLACK ? botB : botW;
      NNResultBuf buf;
      MiscNNInputParams params;
      params.playoutDoublingAdvantage =
        (otherGameProps.playoutDoublingAdvantagePla == C_EMPTY || otherGameProps.playoutDoublingAdvantagePla == pla) ?
        otherGameProps.playoutDoublingAdvantage : -otherGameProps.playoutDoublingAdvantage;
      bot->evaluatePosition(board,hist,pla,params,buf,false);
      finalValueTargets.win = buf.result->whiteWinProb;
      finalValueTargets.draw = buf.result->whiteDrawProb;
      finalValueTargets.loss = buf.result->whiteLossProb;
      finalValueTargets.noResult = 0.0f;
    }
    else {
      // Go retains source move-limit scoring; placement results come only from their rules.
      if(hist.rules.isGo()) hist.endAndScoreGameNow(board);

      finalValueTargets.win = hist.winner == P_WHITE ? 1.0f : 0.0f;
      finalValueTargets.draw = hist.winner == C_EMPTY ? 1.0f : 0.0f;
      finalValueTargets.loss = hist.winner == P_BLACK ? 1.0f : 0.0f;
      finalValueTargets.noResult = 0.0f;

    }
    gameData->whiteValueTargetsByTurn.push_back(finalValueTargets);

    //If we had a hintloc, then don't trust the first value, it will be corrupted a bit by the forced playouts.
    //Just copy the next turn's value.
    if(otherGameProps.hintLoc != Board::NULL_LOC) {
      gameData->whiteValueTargetsByTurn[0] = gameData->whiteValueTargetsByTurn[std::min((size_t)1,gameData->whiteValueTargetsByTurn.size()-1)];
    }


    gameData->hasFullData = true;

    testAssert(gameData->whiteValueTargetsByTurn.size() == gameData->targetWeightByTurn.size() + 1);
    testAssert(rawNNValues.size() == gameData->targetWeightByTurn.size());

    vector<double> valueSurpriseByTurn;
    computeValueSurpriseByTurn(
      valueSurpriseByTurn, gameData->whiteValueTargetsByTurn, rawNNValues, board.x_size * board.y_size,
      playSettings.useSearchValueSurprise
    );

    //After the game, select some of the positions that got only cheap searches and redo them with full searches,
    //so that they get recorded as full training positions. This happens entirely after the game so it cannot
    //affect the game's moves or outcome in any way.
    if(playSettings.useReanalyze && playSettings.reanalyzeProp > 0.0) {
      reanalyzeCheapSearchPositions(
        botB, botW, playSettings, otherGameProps, clearBotBeforeSearch,
        wasCheapSearchByTurn, valueSurpriseByTurn, historicalMctsWinLossValues,
        gameRand, shouldStop, shouldPause, maybeCheckForNewNNEval,
        rawNNValues, locsBuf, playSelectionValuesBuf,
        gameData
      );
      //If we had a hintloc, refresh the copy of turn 1's value into turn 0 made above, in case turn 1 was
      //reanalyzed and its value was updated.
      if(otherGameProps.hintLoc != Board::NULL_LOC) {
        gameData->whiteValueTargetsByTurn[0] = gameData->whiteValueTargetsByTurn[std::min((size_t)1,gameData->whiteValueTargetsByTurn.size()-1)];
      }

      //Reanalysis overwrote the searched values at the reanalyzed turns, which feed into the value targets and
      //the value surprise of every turn at or before them (of just those turns themselves, for search value
      //surprise), so recompute.
      computeValueSurpriseByTurn(
        valueSurpriseByTurn, gameData->whiteValueTargetsByTurn, rawNNValues, board.x_size * board.y_size,
        playSettings.useSearchValueSurprise
      );

      if(logSearchInfo || logMoves) {
        int numCheapSearchTurns = 0;
        for(size_t i = 0; i<wasCheapSearchByTurn.size(); i++)
          numCheapSearchTurns += wasCheapSearchByTurn[i] ? 1 : 0;
        int numReanalyzedTurns = 0;
        for(size_t i = 0; i<gameData->reanalysisByTurn.size(); i++)
          numReanalyzedTurns += gameData->reanalysisByTurn[i].wasReanalyzed ? 1 : 0;
        logger.write(
          "Reanalyzed " + Global::intToString(numReanalyzedTurns) + " of " +
          Global::intToString(numCheapSearchTurns) + " cheap-search positions for training data"
        );
      }
    }

    //Record for statistical purposes (e.g. the selfplaysurprisedump command)
    gameData->valueSurpriseByTurn = valueSurpriseByTurn;
    gameData->wasCheapSearchByTurn = wasCheapSearchByTurn;

    //Compute desired expectation with which to write main game rows
    if(playSettings.policySurpriseDataWeight > 0 || playSettings.valueSurpriseDataWeight > 0) {
      size_t numWeights = gameData->targetWeightByTurn.size();
      testAssert(numWeights == gameData->policySurpriseByTurn.size());

      double sumWeights = 0.0;
      double sumPolicySurpriseWeighted = 0.0;
      double sumValueSurpriseWeighted = 0.0;
      for(size_t i = 0; i < numWeights; i++) {
        float targetWeight = gameData->targetWeightByTurn[i];
        testAssert(targetWeight >= 0.0 && targetWeight <= 1.0);
        sumWeights += targetWeight;
        double policySurprise = gameData->policySurpriseByTurn[i];
        testAssert(policySurprise >= 0.0);
        double valueSurprise = valueSurpriseByTurn[i];
        testAssert(valueSurprise >= 0.0);
        sumPolicySurpriseWeighted += policySurprise * targetWeight;
        sumValueSurpriseWeighted += valueSurprise * targetWeight;
      }

      if(sumWeights >= 1) {
        double averagePolicySurpriseWeighted = sumPolicySurpriseWeighted / sumWeights;
        double averageValueSurpriseWeighted = sumValueSurpriseWeighted / sumWeights;

        //It's possible that we have very little value surprise, such as if the game was initialized lopsided and never again changed
        //from that and the expected player won. So if the total value surprise on targetWeighted turns is too small, then also don't
        //do much valueSurpriseDataWeight, since it would be basically dividing by almost zero, in potentially weird ways.
        double valueSurpriseDataWeight = playSettings.valueSurpriseDataWeight;
        if(averageValueSurpriseWeighted < 0.010) { //0.010 logits on average, pretty arbitrary, mainly just intended limit to extreme cases.
          valueSurpriseDataWeight *= averageValueSurpriseWeighted / 0.010;
        }

        //We also include some rows from non-full searches, if despite the shallow search
        //they were quite surprising to the policy. When using reanalyze, reanalysis of the most surprising
        //cheap searches replaces this behavior for cheap searches that were not reanalyzed, so for those the
        //excess surprise term is turned off. It still applies to all other reduced-weight rows - rows whose
        //visits and weight were reduced due to extreme winrates, including reanalyzed rows reduced that way
        //(judged on the reanalysis search's own policy surprise).
        double thresholdToIncludeReduced = averagePolicySurpriseWeighted * 1.5;
        auto policySurprisePropValueOfTurn = [&](int i) {
          float targetWeight = gameData->targetWeightByTurn[i];
          double policySurprise = gameData->policySurpriseByTurn[i];
          bool excludeExcessSurprise =
            playSettings.useReanalyze && wasCheapSearchByTurn[i] && !gameData->reanalysisByTurn[i].wasReanalyzed;
          double excessSurprise = excludeExcessSurprise ? 0.0 : std::max(0.0,policySurprise-thresholdToIncludeReduced);
          return targetWeight * policySurprise + (1-targetWeight) * excessSurprise;
        };
        auto valueSurprisePropValueOfTurn = [&](int i) {
          float targetWeight = gameData->targetWeightByTurn[i];
          double valueSurprise = valueSurpriseByTurn[i];
          return targetWeight * valueSurprise;
        };

        //Part of the weight will be proportional to surprisePropValue which is just policySurprise on normal rows
        //and the excess policySurprise beyond threshold on shallow searches.
        //First pass - we sum up the surpriseValue.
        double sumPolicySurprisePropValue = 0.0;
        double sumValueSurprisePropValue = 0.0;
        for(int i = 0; i<numWeights; i++) {
          sumPolicySurprisePropValue += policySurprisePropValueOfTurn(i);
          sumValueSurprisePropValue += valueSurprisePropValueOfTurn(i);
        }

        //Just in case, avoid div by 0
        sumPolicySurprisePropValue = std::max(sumPolicySurprisePropValue,1e-10);
        sumValueSurprisePropValue = std::max(sumValueSurprisePropValue,1e-10);

        for(int i = 0; i<numWeights; i++) {
          //Note: the prop value lambdas read targetWeightByTurn[i], so they must be called before we
          //overwrite it below, and each iteration reads only its own index.
          float targetWeight = gameData->targetWeightByTurn[i];
          double policySurprisePropValue = policySurprisePropValueOfTurn(i);
          double valueSurprisePropValue = valueSurprisePropValueOfTurn(i);
          double newValue =
            (1.0-playSettings.policySurpriseDataWeight-valueSurpriseDataWeight) * targetWeight
            + playSettings.policySurpriseDataWeight * policySurprisePropValue * sumWeights / sumPolicySurprisePropValue
            + valueSurpriseDataWeight * valueSurprisePropValue * sumWeights / sumValueSurprisePropValue;
          gameData->targetWeightByTurn[i] = (float)(newValue);
        }
      }
    }

    //Also evaluate all the side positions as well that we queued up to be searched
    NNResultBuf nnResultBuf;
    for(int i = 0; i<sidePositionsToSearch.size(); i++) {
      SidePosition* sp = sidePositionsToSearch[i];

      if(shouldPause != nullptr)
        shouldPause->waitUntilFalse();
      if(shouldStop != nullptr && shouldStop()) {
        delete sp;
        continue;
      }

      Search* toMoveBot = sp->pla == P_BLACK ? botB : botW;
      toMoveBot->setPosition(sp->pla,sp->board,sp->hist);
      //We do NOT apply playoutDoublingAdvantage here. If changing this, note that it is coordinated with train data writing
      //not using playoutDoublingAdvantage for these rows too.
      testAssert(toMoveBot->searchParams.playoutDoublingAdvantage == 0.0);
      testAssert(toMoveBot->searchParams.playoutDoublingAdvantagePla == C_EMPTY);
      sp->playoutDoublingAdvantagePla = C_EMPTY;
      sp->playoutDoublingAdvantage = 0.0;
      Loc responseLoc = toMoveBot->runWholeSearchAndGetMove(sp->pla);

      Play::extractPolicyTarget(sp->policyTarget, toMoveBot, toMoveBot->rootNode, locsBuf, playSelectionValuesBuf);
      extractValueTargets(sp->whiteValueTargets, toMoveBot, toMoveBot->rootNode);
      extractQValueTargets(sp->whiteQValueTargets.targets, toMoveBot, toMoveBot->rootNode);

      double policySurprise = 0.0, policyEntropy = 0.0, searchEntropy = 0.0;
      bool success = toMoveBot->getPolicySurpriseAndEntropy(policySurprise, searchEntropy, policyEntropy);
      testAssert(success);
      (void)success; //Avoid warning when asserts are disabled
      sp->policySurprise = policySurprise;
      sp->policyEntropy = policyEntropy;
      sp->searchEntropy = searchEntropy;

      sp->nnRawStats = computeNNRawStats(toMoveBot, sp->board, sp->hist, sp->pla);
      sp->targetWeight = 1.0f;
      sp->unreducedNumVisits = toMoveBot->getRootVisits();
      sp->numNeuralNetChangesSoFar = (int)gameData->changedNeuralNets.size();

      gameData->sidePositions.push_back(sp);

      //If enabled, also record subtree positions from the search as training positions
      if(playSettings.recordTreePositions && playSettings.recordTreeTargetWeight > 0.0f) {
        if(playSettings.recordTreeTargetWeight > 1.0f)
          throw StringError("playSettings.recordTreeTargetWeight > 1.0f");
        recordTreePositions(
          gameData,
          sp->board,sp->hist,sp->pla,
          toMoveBot,
          playSettings.recordTreeThreshold,playSettings.recordTreeTargetWeight,
          (int)gameData->changedNeuralNets.size(),
          locsBuf,playSelectionValuesBuf,
          Board::NULL_LOC, Board::NULL_LOC
        );
      }

      //Occasionally continue the fork a second move or more, to provide some situations where the opponent has played "weird" moves not
      //only on the most immediate turn, but rather the turns before.
      if(gameRand.nextBool(0.25)) {
        if(responseLoc == Board::NULL_LOC || !sp->hist.isLegal(sp->board,responseLoc,sp->pla))
          failIllegalMove(toMoveBot,logger,sp->board,responseLoc);

        SidePosition* sp2 = new SidePosition(sp->board,sp->hist,sp->pla,(int)gameData->changedNeuralNets.size());
        sp2->hist.makeBoardMoveAssumeLegal(sp2->board,responseLoc,sp2->pla,NULL);
        sp2->pla = getOpp(sp2->pla);
        if(sp2->hist.isGameFinished)
          delete sp2;
        else {
          Search* toMoveBot2 = sp2->pla == P_BLACK ? botB : botW;
          MiscNNInputParams nnInputParams;
          nnInputParams.drawEquivalentWinsForWhite = toMoveBot2->searchParams.drawEquivalentWinsForWhite;
          //Featurize the way this bot's own searches would, even if the game-level history differs.
          nnInputParams.passAliveSuicideRulesOverride =
            toMoveBot2->mainEvaluatorHistoryModes().alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
          nnInputParams.excludeTerritoryAdjAtariOverride =
            toMoveBot2->mainEvaluatorHistoryModes().excludeTerritoryAdjacentToAtari ? 1 : 0;
          toMoveBot2->evaluatePosition(
            sp2->board,sp2->hist,sp2->pla,nnInputParams,
            nnResultBuf,false);
          Loc banMove = Board::NULL_LOC;
          Loc forkLoc = chooseRandomForkingMove(nnResultBuf.result.get(), sp2->board, sp2->hist, sp2->pla, gameRand, banMove);
          if(forkLoc != Board::NULL_LOC) {
            sp2->hist.makeBoardMoveAssumeLegal(sp2->board,forkLoc,sp2->pla,NULL);
            sp2->pla = getOpp(sp2->pla);
            if(sp2->hist.isGameFinished) delete sp2;
            else sidePositionsToSearch.push_back(sp2);
          }
        }
      }

      testAssert(gameData->endHist.moveHistory.size() < 0x1FFFffff);
      maybeCheckForNewNNEval((int)gameData->endHist.moveHistory.size());
    }

    if(playSettings.scaleDataWeight != 1.0) {
      for(int i = 0; i<gameData->targetWeightByTurn.size(); i++)
        gameData->targetWeightByTurn[i] = (float)(playSettings.scaleDataWeight * gameData->targetWeightByTurn[i]);
      for(int i = 0; i<gameData->sidePositions.size(); i++)
        gameData->sidePositions[i]->targetWeight = (float)(playSettings.scaleDataWeight * gameData->sidePositions[i]->targetWeight);
    }

    //Record weights before we possibly probabilistically resolve them
    {
      gameData->targetWeightByTurnUnrounded.resize(gameData->targetWeightByTurn.size());
      for(int i = 0; i<gameData->targetWeightByTurn.size(); i++)
        gameData->targetWeightByTurnUnrounded[i] = gameData->targetWeightByTurn[i];
      for(int i = 0; i<gameData->sidePositions.size(); i++)
        gameData->sidePositions[i]->targetWeightUnrounded = gameData->sidePositions[i]->targetWeight;
    }

    //Resolve probabilistic weights of things
    if(!playSettings.noResolveTargetWeights) {
      auto resolveWeight = [&gameRand](float weight){
        if(weight <= 0) weight = 0;
        float floored = floor(weight);
        float excess = weight - floored;
        weight = gameRand.nextBool(excess) ? floored+1 : floored;
        return weight;
      };

      for(int i = 0; i<gameData->targetWeightByTurn.size(); i++)
        gameData->targetWeightByTurn[i] = resolveWeight(gameData->targetWeightByTurn[i]);
      for(int i = 0; i<gameData->sidePositions.size(); i++)
        gameData->sidePositions[i]->targetWeight = resolveWeight(gameData->sidePositions[i]->targetWeight);
    }



  }

  gameData->trainingWeight = otherGameProps.trainingWeight;
  return gameData;
}

static void replayGameUpToMove(const FinishedGameData* finishedGameData, int moveIdx, Rules rules, Board& board, BoardHistory& hist, Player& pla) {
  board = finishedGameData->startHist.initialBoard;
  pla = finishedGameData->startHist.initialPla;

  //Replay under the same BoardHistoryModes that the game was actually played and
  //adjudicated with (clear() below preserves these).
  hist.setModes(finishedGameData->endHist.modes);

  if(rules.isXiangqi()) {
    // The color-only initialBoard cannot restore pieces or check/chase history.
    hist = finishedGameData->startHist.copyToInitial();
    hist.setModes(finishedGameData->endHist.modes);
  }
  else if(rules.scoringRule == Rules::SCORING_AREA)
    hist.clear(board,pla,rules,0);
  else
    hist.clear(board,pla,rules,finishedGameData->startHist.initialEncorePhase);
  if(!rules.isGo()) hist.setInitialTurnNumber(finishedGameData->startHist.initialTurnNumber);

  //Make sure it's prior to the last move
  if(finishedGameData->endHist.moveHistory.size() <= 0)
    return;
  moveIdx = std::min(moveIdx,(int)(finishedGameData->endHist.moveHistory.size()-1));

  //Replay all those moves
  for(int i = 0; i<moveIdx; i++) {
    Loc loc = finishedGameData->endHist.moveHistory[i].loc;
    if(!hist.isLegal(board,loc,pla)) {
      //We have a bug of some sort if we got an illegal move on replay, unless
      //we are in encore phase (pass for ko may change) or the rules are different
      if(rules == finishedGameData->startHist.rules && hist.encorePhase == 0) {
        cout << board << endl;
        cout << PlayerIO::colorToChar(pla) << endl;
        cout << (rules.isXiangqi() ? Xiangqi::actionToString(Xiangqi::locToAction(loc)) : Location::toString(loc,board)) << endl;
        hist.printDebugInfo(cout,board);
        cout << endl;
        throw StringError("Illegal move when replaying to fork game?");
      }
      //Just break out due to the illegal move and stop the replay here
      return;
    }
    testAssert(finishedGameData->endHist.moveHistory[i].pla == pla);
    hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
    pla = getOpp(pla);

    if(hist.isGameFinished)
      return;
  }
}


void Play::maybeForkGame(
  const FinishedGameData* finishedGameData,
  ForkData* forkData,
  const PlaySettings& playSettings,
  Rand& gameRand,
  Search* bot
) {
  if(forkData == NULL)
    return;
  testAssert(finishedGameData->startHist.initialBoard.pos_hash == finishedGameData->endHist.initialBoard.pos_hash);
  testAssert(finishedGameData->startHist.initialPla == finishedGameData->endHist.initialPla);

  //Just for conceptual simplicity, don't early fork games that started in the encore
  if(finishedGameData->startHist.encorePhase != 0)
    return;
  bool earlyFork = gameRand.nextBool(playSettings.earlyForkGameProb);
  bool lateFork = !earlyFork && playSettings.forkGameProb > 0 ? gameRand.nextBool(playSettings.forkGameProb) : false;
  if(!earlyFork && !lateFork)
    return;

  //Pick a random move to fork from near the start
  int moveIdx;
  if(earlyFork) {
    moveIdx = (int)floor(
      gameRand.nextExponential() * (
        playSettings.earlyForkGameExpectedMoveProp * finishedGameData->startBoard.x_size * finishedGameData->startBoard.y_size
      )
    );
  }
  else if(lateFork) {
    moveIdx = finishedGameData->endHist.moveHistory.size() <= 0 ? 0 : (int)gameRand.nextUInt((uint32_t)finishedGameData->endHist.moveHistory.size());
  }
  else {
    ASSERT_UNREACHABLE;
  }

  Board board;
  Player pla;
  BoardHistory hist;
  replayGameUpToMove(finishedGameData, moveIdx, finishedGameData->startHist.rules, board, hist, pla);
  //Just in case if somehow the game is over now, don't actually do anything
  if(hist.isGameFinished)
    return;

  //Pick a move!
  int policySize = hist.rules.isXiangqi() ? Xiangqi::NUM_ACTIONS : NNPos::MAX_NN_POLICY_SIZE;
  if(playSettings.forkGameMaxChoices > policySize)
    throw StringError("playSettings.forkGameMaxChoices exceeds policy size");
  if(playSettings.earlyForkGameMaxChoices > policySize)
    throw StringError("playSettings.earlyForkGameMaxChoices exceeds policy size");
  int maxChoices = earlyFork ? playSettings.earlyForkGameMaxChoices : playSettings.forkGameMaxChoices;
  if(maxChoices < playSettings.forkGameMinChoices)
    throw StringError("playSettings fork game max choices < playSettings.forkGameMinChoices");

  //Generate a selection of a small random number of choices
  int numChoices = gameRand.nextInt(playSettings.forkGameMinChoices, maxChoices);
  testAssert(numChoices <= policySize);
  SearchPolicyBuffer<Loc> possibleMoveStorage(policySize);
  Loc* possibleMoves = possibleMoveStorage.data();
  int numPossible = PlayUtils::chooseRandomLegalMoves(board,hist,pla,gameRand,possibleMoves,numChoices);
  if(numPossible <= 0)
    return;

  //Try the one the value net thinks is best
  Loc bestMove = Board::NULL_LOC;
  double bestWinLoss = 0.0;

  NNResultBuf buf;
  double drawEquivalentWinsForWhite = 0.5;
  for(int i = 0; i<numChoices; i++) {
    Loc loc = possibleMoves[i];
    Board copy = board;
    BoardHistory copyHist = hist;
    copyHist.makeBoardMoveAssumeLegal(copy,loc,pla,NULL);
    double whiteWinLoss;
    if(!copyHist.rules.isGo() && copyHist.isGameFinished) {
      whiteWinLoss = copyHist.winner == P_WHITE ? 1.0 : copyHist.winner == P_BLACK ? -1.0 : 0.0;
    }
    else {
      MiscNNInputParams nnInputParams;
      nnInputParams.drawEquivalentWinsForWhite = drawEquivalentWinsForWhite;
      //Featurize the way this bot's own searches would, even if the replayed game-level history differs.
      nnInputParams.passAliveSuicideRulesOverride =
        bot->mainEvaluatorHistoryModes().alwaysComputePassAliveUnderSuicideRules ? 1 : 0;
      nnInputParams.excludeTerritoryAdjAtariOverride =
        bot->mainEvaluatorHistoryModes().excludeTerritoryAdjacentToAtari ? 1 : 0;
      bot->evaluatePosition(copy,copyHist,getOpp(pla),nnInputParams,buf,false);
      std::shared_ptr<NNOutput> nnOutput = std::move(buf.result);
      whiteWinLoss = nnOutput->whiteWinProb - nnOutput->whiteLossProb;
    }
    if(bestMove == Board::NULL_LOC || (pla == P_WHITE && whiteWinLoss > bestWinLoss) || (pla == P_BLACK && whiteWinLoss < bestWinLoss)) {
      bestMove = loc;
      bestWinLoss = whiteWinLoss;
    }
  }

  //Make that move
  testAssert(hist.isLegal(board,bestMove,pla));
  hist.makeBoardMoveAssumeLegal(board,bestMove,pla,NULL);
  pla = getOpp(pla);

  //If the game is over now, don't actually do anything
  if(hist.isGameFinished)
    return;
  forkData->add(new InitialPosition(board,hist,pla,true,false,finishedGameData->trainingWeight));
}


void Play::maybeHintForkGame(
  const FinishedGameData* finishedGameData,
  ForkData* forkData,
  const OtherGameProperties& otherGameProps
) {
  if(forkData == NULL)
    return;
  //Just for conceptual simplicity, don't early fork games that started in the encore
  if(finishedGameData->startHist.encorePhase != 0)
    return;
  bool hintFork =
    otherGameProps.hintLoc != Board::NULL_LOC &&
    hintPositionHash(finishedGameData->startBoard,finishedGameData->startHist) == otherGameProps.hintPosHash &&
    finishedGameData->startHist.moveHistory.size() == otherGameProps.hintTurn &&
    finishedGameData->endHist.moveHistory.size() > finishedGameData->startHist.moveHistory.size() &&
    finishedGameData->endHist.moveHistory[finishedGameData->startHist.moveHistory.size()].loc != otherGameProps.hintLoc;

  if(!hintFork)
    return;

  Board board;
  Player pla;
  BoardHistory hist;
  testAssert(finishedGameData->startHist.moveHistory.size() < 0x1FFFffff);
  int moveIdxToReplayTo = (int)finishedGameData->startHist.moveHistory.size();
  replayGameUpToMove(finishedGameData, moveIdxToReplayTo, finishedGameData->startHist.rules, board, hist, pla);
  //Just in case if somehow the game is over now, don't actually do anything
  if(hist.isGameFinished)
    return;

  testAssert(pla == hist.presumedNextMovePla);
  if(!hist.isLegal(board,otherGameProps.hintLoc,pla))
    return;

  hist.makeBoardMoveAssumeLegal(board,otherGameProps.hintLoc,pla,NULL);
  pla = getOpp(pla);

  //If the game is over now, don't actually do anything
  if(hist.isGameFinished)
    return;
  forkData->add(new InitialPosition(board,hist,pla,false,true,finishedGameData->trainingWeight));
}


GameRunner::GameRunner(ConfigParser& cfg, const PlaySettings& pSettings, Logger& logger)
  :logSearchInfo(),logMoves(),maxMovesPerGame(),clearBotBeforeSearch(),
   playSettings(pSettings),
   gameInit(NULL)
{
  logSearchInfo = cfg.getBool("logSearchInfo");
  logMoves = cfg.getBool("logMoves");
  maxMovesPerGame = cfg.getInt("maxMovesPerGame",0,1 << 30);
  clearBotBeforeSearch = cfg.contains("clearBotBeforeSearch") ? cfg.getBool("clearBotBeforeSearch") : false;

  //Initialize object for randomizing game settings
  gameInit = new GameInitializer(cfg,logger);
}
GameRunner::GameRunner(ConfigParser& cfg, const string& gameInitRandSeed, const PlaySettings& pSettings, Logger& logger)
  :logSearchInfo(),logMoves(),maxMovesPerGame(),clearBotBeforeSearch(),
   playSettings(pSettings),
   gameInit(NULL)
{
  logSearchInfo = cfg.getBool("logSearchInfo");
  logMoves = cfg.getBool("logMoves");
  maxMovesPerGame = cfg.getInt("maxMovesPerGame",0,1 << 30);
  clearBotBeforeSearch = cfg.contains("clearBotBeforeSearch") ? cfg.getBool("clearBotBeforeSearch") : false;

  //Initialize object for randomizing game settings
  gameInit = new GameInitializer(cfg,logger,gameInitRandSeed);
}

GameRunner::~GameRunner() {
  delete gameInit;
}

const GameInitializer* GameRunner::getGameInitializer() const {
  return gameInit;
}

FinishedGameData* GameRunner::runGame(
  const string& seed,
  const MatchPairer::BotSpec& bSpecB,
  const MatchPairer::BotSpec& bSpecW,
  ForkData* forkData,
  const Sgf::PositionSample* startPosSample,
  Logger& logger,
  const std::function<bool()>& shouldStop,
  const WaitableFlag* shouldPause,
  const std::function<NNEvaluator*()>& checkForNewNNEval,
  const std::function<void(const MatchPairer::BotSpec&, Search*)>& afterInitialization,
  const std::function<void(const Board&, const BoardHistory&, Player, Loc, const std::vector<double>&, const Search*)>& onEachMove
) {
  MatchPairer::BotSpec botSpecB = bSpecB;
  MatchPairer::BotSpec botSpecW = bSpecW;

  Rand gameRand(seed + ":" + "forGameRand");

  const InitialPosition* initialPosition = NULL;
  if(forkData != NULL) {
    initialPosition = forkData->get(gameRand);

  }

  Board board;
  Player pla;
  BoardHistory hist;
  int numExtraBlack = 0;
  OtherGameProperties otherGameProps;
  //Game-level BoardHistoryModes for this game, matching the values Play::runGame will
  //stamp onto the game history it plays out: each on only if BOTH bots' searches will be using it.
  const BoardHistoryModes modesB = botSpecB.historyModes();
  const BoardHistoryModes modesW = botSpecW.historyModes();
  const BoardHistoryModes gameHistoryModes = BoardHistoryModes(
    modesB.alwaysComputePassAliveUnderSuicideRules && modesW.alwaysComputePassAliveUnderSuicideRules,
    modesB.excludeTerritoryAdjacentToAtari && modesW.excludeTerritoryAdjacentToAtari
  );
  if(playSettings.forSelfPlay) {
    testAssert(botSpecB.botIdx == botSpecW.botIdx);
    SearchParams params = botSpecB.baseParams;
    gameInit->createGame(board,pla,hist,numExtraBlack,params,initialPosition,playSettings,otherGameProps,startPosSample,gameHistoryModes);
    botSpecB.baseParams = params;
    botSpecW.baseParams = params;
  }
  else {
    gameInit->createGame(board,pla,hist,numExtraBlack,initialPosition,playSettings,otherGameProps,startPosSample,gameHistoryModes);

    bool rulesWereSupported;
    if(botSpecB.nnEval != NULL) {
      botSpecB.nnEval->getSupportedRules(hist.rules,rulesWereSupported);
      if(!rulesWereSupported)
        logger.write("WARNING: Match is running bot on rules that it does not support: " + botSpecB.botName);
    }
    if(botSpecW.nnEval != NULL) {
      botSpecW.nnEval->getSupportedRules(hist.rules,rulesWereSupported);
      if(!rulesWereSupported)
        logger.write("WARNING: Match is running bot on rules that it does not support: " + botSpecW.botName);
    }
  }

  bool clearBotBeforeSearchThisGame = clearBotBeforeSearch;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    //Avoid interactions between the two bots since they're the same.
    //Also in self-play this makes sure root noise is effective on each new search
    clearBotBeforeSearchThisGame = true;
  }

  //In 2% of games, don't autoterminate the game upon all pass alive, to just provide a tiny bit of training data on positions that occur
  //as both players must wrap things up manually, because within the search we don't autoterminate games, meaning that the NN will get
  //called on positions that occur after the game would have been autoterminated.
  bool doEndGameIfAllPassAlive = playSettings.forSelfPlay ? gameRand.nextBool(0.98) : true;

  Search* botB;
  Search* botW;
  if(botSpecB.botIdx == botSpecW.botIdx) {
    botB = botSpecB.createSearch(logger,seed);
    botW = botB;
  }
  else {
    botB = botSpecB.createSearch(logger,seed + "@B");
    botW = botSpecW.createSearch(logger,seed + "@W");
  }
  if(afterInitialization != nullptr) {
    if(botSpecB.botIdx == botSpecW.botIdx) {
      afterInitialization(botSpecB,botB);
    }
    else {
      afterInitialization(botSpecB,botB);
      afterInitialization(botSpecW,botW);
    }
  }

  FinishedGameData* finishedGameData = Play::runGame(
    board,pla,hist,numExtraBlack,
    botSpecB,botSpecW,
    botB,botW,
    doEndGameIfAllPassAlive,clearBotBeforeSearchThisGame,
    logger,logSearchInfo,logMoves,
    maxMovesPerGame,shouldStop,
    shouldPause,
    playSettings,otherGameProps,
    gameRand,
    checkForNewNNEval, //Note that if this triggers, botSpecB and botSpecW will get updated, for use in maybeForkGame
    onEachMove
  );

  if(initialPosition != NULL) {
    finishedGameData->usedInitialPosition = 1;
    testAssert(finishedGameData->trainingWeight == initialPosition->trainingWeight);
  }
  else if(startPosSample != NULL) {
    testAssert(finishedGameData->trainingWeight == startPosSample->trainingWeight);
  }

  testAssert(finishedGameData->trainingWeight > 0.0);
  testAssert(finishedGameData->trainingWeight < 5.0);

  //Make sure not to write the game if we terminated in the middle of this game!
  if(shouldStop != nullptr && shouldStop()) {
    if(botW != botB)
      delete botW;
    delete botB;
    delete finishedGameData;
    return NULL;
  }

  testAssert(finishedGameData != NULL);

  Play::maybeForkGame(finishedGameData, forkData, playSettings, gameRand, botB);
  Play::maybeHintForkGame(finishedGameData, forkData, otherGameProps);

  if(botW != botB)
    delete botW;
  delete botB;

  if(initialPosition != NULL)
    delete initialPosition;

  return finishedGameData;
}
