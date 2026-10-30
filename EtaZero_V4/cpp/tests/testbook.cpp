#include "../tests/tests.h"

#include <algorithm>
#include <iterator>
#include <iomanip>

#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../book/book.h"
#include "../game/xiangqi.h"
#include "../search/mutexpool.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;
using namespace TestSearchCommon;

namespace {
BookHash gameBookHash(const BoardHistory& hist) {
  BookHash hash;
  int symmetry;
  vector<int> stabilizers;
  BookHash::getHashAndSymmetry(hist,9,hash,symmetry,stabilizers,Book::LATEST_BOOK_VERSION);
  testAssert(hist.rules.allowsSymmetry(symmetry));
  for(int s: stabilizers) testAssert(hist.rules.allowsSymmetry(s));
  return hash;
}

unique_ptr<Book> makeGameBook(const BoardHistory& hist) {
  auto book = make_unique<Book>(Book::LATEST_BOOK_VERSION,hist.getRecentBoard(0),hist.rules,hist.presumedNextMovePla,
    9,hist.modes,BookParams(),hist.rules.isXiangqi() ? &hist.getXiangqiState() : nullptr,hist.getCurrentTurnNumber());
  // As in genbook, evaluate the root before adding children. The source cost
  // heuristic compares expanded visits against visits of the unexpanded part.
  auto& values = book->getRoot().thisValuesNotInBook();
  values.winLossValue = 0.125;
  values.winLossError = 0.03;
  values.maxPolicy = 0.1;
  values.visits = values.weight = 10;
  return book;
}

template<typename F> void expectBookError(F f) {
  bool rejected = false;
  try { f(); } catch(const std::exception&) { rejected = true; }
  testAssert(rejected);
}

void checkGameBook(Book& book) {
  for(SymBookNode node: book.getAllNodes()) {
    for(int sym = 0; sym < SymmetryHelpers::NUM_SYMMETRIES; sym++) {
      if(!book.allowsSymmetry(sym)) {
        expectBookError([&]() { node.applySymmetry(sym); });
        expectBookError([&]() { ConstSymBookNode(node).applySymmetry(sym); });
        continue;
      }
      ConstSymBookNode oriented = node.applySymmetry(sym);
      BoardHistory hist;
      vector<Loc> moves;
      testAssert(oriented.getBoardHistoryReachingHere(hist,moves));
      testAssert(gameBookHash(hist) == node.hash());
      testAssert(hist.rules == book.initialRules && hist.modes == book.historyModes);
      for(const BookMove& move: oriented.getUniqueMovesInBook()) {
        testAssert(book.parseMove(book.moveToString(move.move)) == move.move);
        testAssert(hist.isLegal(hist.getRecentBoard(0),move.move,hist.presumedNextMovePla));
        Board board = hist.getRecentBoard(0);
        BoardHistory childHist = hist;
        childHist.makeBoardMoveAssumeLegal(board,move.move,childHist.presumedNextMovePla,nullptr);
        testAssert(gameBookHash(childHist) == oriented.follow(move.move).hash());
      }
    }
  }
}

unique_ptr<Book> roundTripGameBook(Book& book) {
  book.recomputeEverything();
  ostringstream out;
  book.saveToStream(out);
  istringstream in(out.str());
  unique_ptr<Book> loaded(Book::loadFromStream(in,2));
  testAssert(loaded->size() == book.size());
  testAssert(loaded->initialRules == book.initialRules && loaded->historyModes == book.historyModes);
  testAssert(loaded->initialBoard.isEqualForTesting(book.initialBoard,false,false,!book.initialRules.isGo()));
  testAssert(loaded->getInitialHist().getCurrentTurnNumber() == book.getInitialHist().getCurrentTurnNumber());
  testAssert(loaded->getRoot().hash() == book.getRoot().hash());
  if(book.initialRules.isXiangqi()) {
    testAssert(loaded->initialXiangqiState->record().initialFen == book.initialXiangqiState->record().initialFen);
    testAssert(loaded->initialXiangqiState->record().actions == book.initialXiangqiState->record().actions);
  }
  checkGameBook(*loaded);
  for(SymBookNode node: book.getAllNodes()) {
    SymBookNode other = loaded->getByHash(node.hash());
    testAssert(!other.isNull());
    if(!(abs(node.totalExpansionCost()-other.totalExpansionCost()) < 1e-4))
      cerr << "Book cost mismatch: " << Rules::gameName(book.initialRules.game) << " " << node.hash()
           << " " << node.totalExpansionCost() << " vs " << other.totalExpansionCost() << endl;
    testAssert(abs(node.totalExpansionCost()-other.totalExpansionCost()) < 1e-4);
    testAssert(abs(node.recursiveValues().winLossValue-other.recursiveValues().winLossValue) < 1e-4);
  }
  return loaded;
}

SymBookNode addBookMove(Book& book, SymBookNode node, Board& board, BoardHistory& hist, Loc move) {
  SymBookNode child;
  if(node.isMoveInBook(move)) child = node.playMove(board,hist,move);
  else {
    bool transposing;
    child = node.playAndAddMove(board,hist,move,0.125,transposing);
  }
  testAssert(!child.isNull() && child.hash() == gameBookHash(hist));
  BoardHistory replayed;
  vector<Loc> moves;
  testAssert(child.getBoardHistoryReachingHere(replayed,moves));
  testAssert(replayed.getRecentBoard(0).pos_hash == board.pos_hash);
  if(hist.rules.isXiangqi())
    testAssert(replayed.getXiangqiState().historyHash() == hist.getXiangqiState().historyHash());
  auto& values = child.thisValuesNotInBook();
  values.winLossValue = 0.125;
  values.winLossError = 0.03;
  values.maxPolicy = 0.1;
  values.visits = values.weight = 10;
  testAssert(!book.getByHash(child.hash()).isNull());
  return child;
}
}

void Tests::runBookGameTests(const string& outputDir) {
  cout << "Running multi-game book tests" << endl;
  using G = Rules::Game;
  Rand rand("multiGameBook");
  set<BookHash> identities;
  for(G game: {G::GO,G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX,G::XIANGQI}) {
    const Rules rules = Rules::forGame(game);
    BoardHistory initial;
    if(rules.isXiangqi()) initial.clear(Xiangqi::State());
    else initial.clear(Board(7,7),P_BLACK,rules,0);
    initial.setInitialTurnNumber(rules.isGo() ? 0 : 17);
    auto book = makeGameBook(initial);
    testAssert(identities.insert(book->getRoot().hash()).second);
    auto syms = book->getRoot().getSymmetries();
    if(game == G::HEX) testAssert(syms == vector<int>({0,3}));
    else if(game == G::XIANGQI) testAssert(syms == vector<int>({0,2}));
    else testAssert(syms.size() == 8);
    for(int branch = 0; branch < 8; branch++) {
      BoardHistory hist = book->getInitialHist();
      Board board = hist.getRecentBoard(0);
      auto node = book->getRoot();
      for(int depth = 0; depth < 6 && !hist.isGameFinished; depth++) {
        vector<Loc> legal;
        if(rules.isXiangqi()) {
          for(int a: hist.getXiangqiState().legalActions()) legal.push_back(Xiangqi::actionToLoc(a));
        }
        else for(int y = 0; y < board.y_size; y++) for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x,y,board.x_size);
          if(hist.isLegal(board,loc,hist.presumedNextMovePla)) legal.push_back(loc);
        }
        testAssert(!legal.empty());
        node = addBookMove(*book,node,board,hist,legal[rand.nextUInt((uint32_t)legal.size())]);
      }
    }
    checkGameBook(*book);
    auto loaded = roundTripGameBook(*book);
    if(!outputDir.empty()) {
      ofstream out;
      FileUtils::open(out,outputDir + "/" + Rules::gameName(game) + ".katabook");
      loaded->saveToStream(out);
    }
  }

  // Legal placements with no Go liberties must survive symmetry and file I/O.
  for(G game: {G::FREESTYLE,G::STANDARD,G::RENJU,G::HEX}) {
    Board board(7,7);
    for(auto xy: {pair<int,int>{2,3},{4,3},{3,2},{3,4}})
      board.colors[Location::getLoc(xy.first,xy.second,7)] = P_WHITE;
    board.colors[Location::getLoc(3,3,7)] = P_BLACK;
    board.regenChainsFromColors();
    BoardHistory initial(board,P_BLACK,Rules::forGame(game),0,BoardHistoryModes());
    auto book = makeGameBook(initial);
    for(int sym = 0; sym < 8; sym++) if(book->allowsSymmetry(sym)) {
      auto hist = book->getInitialHist(sym);
      const auto& b = hist.getRecentBoard(0);
      testAssert(b.numStonesOnBoard() == 5 && b.getNumLiberties(Location::getLoc(3,3,7)) == 0);
      testAssert(gameBookHash(hist) == book->getRoot().hash());
    }
    roundTripGameBook(*book);

    // Move ordering can transpose in monotonic placement games.
    Board a(7,7), b(7,7);
    BoardHistory ha(a,P_BLACK,initial.rules,0,BoardHistoryModes()), hb = ha;
    for(auto xy: {pair<int,int>{0,0},{4,4},{1,0},{5,4}})
      ha.makeBoardMoveAssumeLegal(a,Location::getLoc(xy.first,xy.second,7),ha.presumedNextMovePla,nullptr);
    for(auto xy: {pair<int,int>{1,0},{5,4},{0,0},{4,4}})
      hb.makeBoardMoveAssumeLegal(b,Location::getLoc(xy.first,xy.second,7),hb.presumedNextMovePla,nullptr);
    testAssert(gameBookHash(ha) == gameBookHash(hb));

    // A real winning move followed by a terminal node, for startpose export.
    if(!outputDir.empty()) {
      Board before(5,5);
      for(int y = 0; y < 4; y++) before.colors[Location::getLoc(0,y,5)] = P_BLACK;
      before.colors[Location::getLoc(4,0,5)] = before.colors[Location::getLoc(4,2,5)] = P_WHITE;
      before.regenChainsFromColors();
      BoardHistory ending(before,P_BLACK,initial.rules,0,BoardHistoryModes());
      ending.setInitialTurnNumber(41);
      testAssert(!ending.isGameFinished);
      auto terminalBook = makeGameBook(ending);
      addBookMove(*terminalBook,terminalBook->getRoot(),before,ending,Location::getLoc(0,4,5));
      testAssert(ending.isGameFinished && ending.winner == P_BLACK);
      terminalBook->recomputeEverything();
      ofstream out;
      FileUtils::open(out,outputDir + "/" + Rules::gameName(game) + "-terminal.katabook");
      terminalBook->saveToStream(out);
    }
  }

  // Hex's two diagonals have different connectivity. Reflection must not merge them.
  Board a(3,3), b(3,3);
  for(int y = 0; y < 3; y++) {
    a.colors[Location::getLoc(y,y,3)] = P_BLACK;
    b.colors[Location::getLoc(2-y,y,3)] = P_BLACK;
  }
  a.regenChainsFromColors(); b.regenChainsFromColors();
  testAssert(gameBookHash(BoardHistory(a,P_WHITE,Rules::forGame(G::HEX),0,BoardHistoryModes())) !=
    gameBookHash(BoardHistory(b,P_WHITE,Rules::forGame(G::HEX),0,BoardHistoryModes())));

  // The origin includes earlier rule history even when the final board is unchanged.
  Xiangqi::State prior;
  for(const char* move: {"b0c2","b9c7","c2b0","c7b9"}) prior.play(Xiangqi::parseAction(move));
  BoardHistory initial(prior,BoardHistoryModes(true,true));
  initial.setInitialTurnNumber(97);
  auto book = makeGameBook(initial);
  Xiangqi::State fresh(prior.fen());
  testAssert(book->getRoot().hash() != gameBookHash(BoardHistory(fresh,initial.modes)));
  testAssert(gameBookHash(book->getInitialHist(2)) == book->getRoot().hash());
  auto loaded = roundTripGameBook(*book);
  BoardHistory hist = loaded->getInitialHist();
  Board board = hist.getRecentBoard(0);
  auto node = loaded->getRoot();
  for(const char* move: {"b0c2","b9c7","c2b0","c7b9"})
    node = addBookMove(*loaded,node,board,hist,Xiangqi::actionToLoc(Xiangqi::parseAction(move)));
  testAssert(hist.isGameFinished && hist.winner == C_EMPTY && hist.getCurrentTurnNumber() == 101);
  testAssert(hist.getXiangqiState().result().reason == Xiangqi::Reason::REPETITION);
  roundTripGameBook(*loaded);
  expectBookError([&]() { Book invalid(Book::LATEST_BOOK_VERSION,board,initial.rules,P_BLACK,9,initial.modes,BookParams()); });
  if(!outputDir.empty()) {
    ofstream out;
    FileUtils::open(out,outputDir + "/xiangqi-history.katabook");
    loaded->saveToStream(out);
  }

  struct HistoryCase {
    string name;
    string fen;
    vector<string> cycle;
  };
  string limitFen = Xiangqi::State::START_FEN;
  limitFen.replace(limitFen.rfind("0 1"),3,"119 61");
  for(const auto& c: vector<HistoryCase>{
      {"check","5k3/4R4/9/9/4P4/9/9/9/9/4K4 w - - 0 1",{"e8f8","f9e9","f8e8","e9f9"}},
      {"chase","5k3/4n4/9/1R7/4P4/9/9/9/9/4K4 w - - 0 1",{"b6b8","e8d6","b8b6","d6e8"}},
      {"limit",limitFen,{"b0c2"}}}) {
    Xiangqi::State state(c.fen);
    if(c.name != "limit") for(const string& move: c.cycle) state.play(Xiangqi::parseAction(move));
    auto historyBook = makeGameBook(BoardHistory(state,BoardHistoryModes()));
    auto restored = roundTripGameBook(*historyBook);
    BoardHistory continuing = restored->getInitialHist();
    Board current = continuing.getRecentBoard(0);
    auto currentNode = restored->getRoot();
    for(const string& move: c.cycle)
      currentNode = addBookMove(*restored,currentNode,current,continuing,Xiangqi::actionToLoc(Xiangqi::parseAction(move)));
    testAssert(continuing.isGameFinished && !continuing.isNoResult);
    testAssert(continuing.winner == (c.name == "limit" ? C_EMPTY : P_WHITE));
    testAssert(continuing.getXiangqiState().result().reason ==
      (c.name == "limit" ? Xiangqi::Reason::MOVE_LIMIT : Xiangqi::Reason::REPETITION));
    roundTripGameBook(*restored);
    if(!outputDir.empty()) {
      ofstream out;
      FileUtils::open(out,outputDir + "/xiangqi-" + c.name + ".katabook");
      restored->saveToStream(out);
    }
  }

  auto rectangular = makeGameBook(BoardHistory(Board(4,5),P_BLACK,Rules::getTrompTaylorish(),0,BoardHistoryModes()));
  testAssert(rectangular->getRoot().getSymmetries().size() == 4);
  for(int sym = 0; sym < 8; sym++) {
    if(sym < 4) testAssert(gameBookHash(rectangular->getInitialHist(sym)) == rectangular->getRoot().hash());
    else expectBookError([&]() { rectangular->getInitialHist(sym); });
  }
  cout << "Multi-game book tests passed" << endl;
}

static void corruptNodes(Book* book, std::vector<SymBookNode>& newAndChangedNodes, Rand& rand) {
  for(SymBookNode node: newAndChangedNodes) {
    node.getNodeForTesting()->corruptRecursiveValuesOnlyForTesting(rand);
  }
  for(SymBookNode node: book->getAllNodes()) {
    node.getNodeForTesting()->corruptNodeCostsOnlyForTesting(rand);
  }
}


void Tests::runBookTests() {
  cout << "Running book tests" << endl;

  Board initialBoard(4,4);
  Rules rules = Rules::parseRules("japanese");
  Player initialPla = P_BLACK;

  int repBound = 9;
  BookParams params;
  params.errorFactor = 1.05;
  params.costPerMove = 0.53;
  params.costPerUCBWinLossLoss = 2.04;
  params.costPerUCBWinLossLossPow3 = 0.71;
  params.costPerUCBWinLossLossPow7 = 0.67;
  params.costPerLogPolicy = 0.46;
  params.costPerMovesExpanded = 0.230;
  params.costPerSquaredMovesExpanded = 0.041;
  params.bonusPerWinLossError = 0.32;
  params.bonusPerExcessUnexpandedPolicy = 1.14;
  params.bonusPerUnexpandedBestWinLoss = 1.23;
  params.bonusForWLPV1 = 0.14;
  params.bonusForWLPV2 = 0.17;
  params.bonusForBiggestWLCost = 0.59;
  params.earlyBookCostReductionFactor = 0.52;
  params.earlyBookCostReductionLambda = 0.33;
  params.policyBoostSoftUtilityScale = 0.034;
  params.utilityPerPolicyForSorting = 0.021;
  params.adjustedVisitsWLScale = 0.05;
  params.maxVisitsForReExpansion = 25;
  params.visitsScale = 50;

  string testFileName = "./_test.katabook.tmp";

  Book* book = new Book(
    Book::LATEST_BOOK_VERSION,
    initialBoard,
    rules,
    initialPla,
    repBound,
    BoardHistoryModes(),
    params
  );

  Rand rand("runBookTests");
  Rand corruptionRand("runBookTestsNoise");
  std::vector<Loc> legalMovesBuf;

  MutexPool mutexPool(4096);
  int numThreads = 4;

  for(int waveIdx = 0; waveIdx<35; waveIdx++) {
    cout << "Book size " << book->size() << endl;
    std::set<BookHash> nodesHashesToUpdate;

    for(int branchIdx = 0; branchIdx<10; branchIdx++) {
      SymBookNode node = book->getRoot();
      Board board(initialBoard);
      BoardHistory hist(board,initialPla,rules,0,book->historyModes);
      int branchLen = rand.nextUInt(10);

      for(int turnIdx = 0; turnIdx<branchLen; turnIdx++) {
        testAssert(!node.isNull());
        if(hist.isGameFinished || hist.encorePhase != 0)
          break;

        Player pla = hist.presumedNextMovePla;
        legalMovesBuf.clear();
        for(int y = 0; y<board.y_size; y++) {
          for(int x = 0; x<board.x_size; x++) {
            Loc loc = Location::getLoc(x,y,board.x_size);
            if(hist.isLegal(board,loc,pla))
              legalMovesBuf.push_back(loc);
          }
        }
        {
          Loc loc = Board::PASS_LOC;
          if(hist.isLegal(board,loc,pla))
            legalMovesBuf.push_back(loc);
        }

        Loc moveLoc = legalMovesBuf[rand.nextUInt((uint32_t)legalMovesBuf.size())];
        if(!node.isMoveInBook(moveLoc)) {
          bool childIsTransposing;
          double moveLocPolicy = rand.nextDouble();
          SymBookNode child = node.playAndAddMove(board,hist,moveLoc,moveLocPolicy,childIsTransposing);
          if(!child.isNull() && !childIsTransposing)
            nodesHashesToUpdate.insert(child.hash());
          nodesHashesToUpdate.insert(node.hash());
          node = child;
        }
        else {
          node = node.playMove(board,hist,moveLoc);
        }
      }
    }

    std::vector<SymBookNode> newAndChangedNodes;
    for(BookHash hash: nodesHashesToUpdate) {
      SymBookNode node = book->getByHash(hash);

      BookValues& nodeValues = node.thisValuesNotInBook();
      nodeValues.winLossValue = rand.nextDouble(-1,1);
      nodeValues.winLossError = rand.nextExponential() * 0.1;

      nodeValues.maxPolicy = rand.nextExponential() * 0.1;
      nodeValues.weight = rand.nextDouble(10,20);
      nodeValues.visits = rand.nextDouble(10,20);

      newAndChangedNodes.push_back(node);
    }

    std::map<BookHash,double> costsAfterFirst;
    std::map<BookHash,double> costsAfterSecond;



    if(waveIdx % 2 == 0) {
      //Even waves: recompute then recomputeMultiThreaded

      if(waveIdx % 3 == 0)
        corruptNodes(book, newAndChangedNodes, corruptionRand);

      book->recompute(newAndChangedNodes);
      for(SymBookNode node: book->getAllNodes())
        costsAfterFirst[node.hash()] = node.totalExpansionCost();

      if(waveIdx % 3 == 0)
        corruptNodes(book, newAndChangedNodes, corruptionRand);

      book->recomputeMultiThreaded(newAndChangedNodes, mutexPool, numThreads);
      for(SymBookNode node: book->getAllNodes())
        costsAfterSecond[node.hash()] = node.totalExpansionCost();
    }
    else {
      //Odd waves: recomputeMultiThreaded then recompute

      if(waveIdx % 3 == 0)
        corruptNodes(book, newAndChangedNodes, corruptionRand);

      book->recomputeMultiThreaded(newAndChangedNodes, mutexPool, numThreads);
      for(SymBookNode node: book->getAllNodes())
        costsAfterFirst[node.hash()] = node.totalExpansionCost();

      if(waveIdx % 3 == 0)
        corruptNodes(book, newAndChangedNodes, corruptionRand);

      book->recompute(newAndChangedNodes);
      for(SymBookNode node: book->getAllNodes())
        costsAfterSecond[node.hash()] = node.totalExpansionCost();
    }

    //Should give identical costs
    for(SymBookNode node: book->getAllNodes()) {
      BookHash hash = node.hash();
      testAssert(abs(costsAfterFirst[hash] - costsAfterSecond[hash]) < 1e-5);
    }

    std::map<BookHash,double> costsAfterEverything;
    std::vector<SymBookNode> allSymNodes = book->getAllNodes();
    if(waveIdx % 2 == 0) {

      if(waveIdx % 3 == 0)
        corruptNodes(book, allSymNodes, corruptionRand);

      book->recomputeEverything();
      for(SymBookNode node: book->getAllNodes())
        costsAfterEverything[node.hash()] = node.totalExpansionCost();

      if(waveIdx % 3 == 0)
        corruptNodes(book, allSymNodes, corruptionRand);

      book->recomputeEverythingMultiThreaded(mutexPool, numThreads);

      // Also idempotent here
      for(SymBookNode node: book->getAllNodes()) {
        BookHash hash = node.hash();
        testAssert(abs(costsAfterEverything[hash] - node.totalExpansionCost()) < 1e-5);
        testAssert(abs(costsAfterSecond[hash] - node.totalExpansionCost()) < 1e-5);
      }
    }
    else {

      if(waveIdx % 3 == 0)
        corruptNodes(book, allSymNodes, corruptionRand);

      book->recomputeEverythingMultiThreaded(mutexPool, numThreads);
      for(SymBookNode node: book->getAllNodes())
        costsAfterEverything[node.hash()] = node.totalExpansionCost();

      if(waveIdx % 3 == 0)
        corruptNodes(book, allSymNodes, corruptionRand);

      book->recomputeEverything();

      for(SymBookNode node: book->getAllNodes()) {
        BookHash hash = node.hash();
        testAssert(abs(costsAfterEverything[hash] - node.totalExpansionCost()) < 1e-3);
        testAssert(abs(costsAfterSecond[hash] - node.totalExpansionCost()) < 1e-3);
      }
    }

    Book* loaded = NULL;
    if(waveIdx % 20 == 0) {
      book->saveToFile(testFileName);
      loaded = Book::loadFromFile(testFileName);
      for(SymBookNode node: loaded->getAllNodes()) {
        BookHash hash = node.hash();
        testAssert(abs(costsAfterSecond[hash] - node.totalExpansionCost()) < 1e-3);
      }
      delete loaded;
    }
  }

  const bool logToStdout = false;
  const bool logToStderr = false;
  const bool logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);
  logger.addOStream(cout);
  MakeDir::make("./tests/");
  MakeDir::make("./tests/results/");
  MakeDir::make("./tests/results/bookhtml/");
  string htmlDir = "./tests/results/bookhtml/";
  string rulesLabel = "";
  string rulesLink = "";
  int64_t htmlMinVisits = 200;
  bool htmlDevMode = false;
  book->exportToHtmlDir(htmlDir,rulesLabel,rulesLink,htmlDevMode,htmlMinVisits,logger);

  delete book;
  FileUtils::tryRemoveFile(testFileName);
  cout << "Done" << endl;
}
