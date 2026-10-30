#include "tests.h"
#include "../program/play.h"
#include "../program/setup.h"
#include "../program/selfplaymanager.h"
#include "../search/policybuffer.h"
#include "../core/makedir.h"
#include "../external/nlohmann_json/json.hpp"
#include <fstream>

void Tests::runMuZeroPlay(const std::string& directory, const std::string& configFile, const std::string& output) {
  using json = nlohmann::json;
  Board::initHash();
  ConfigParser config(configFile);
  Logger logger(nullptr,false,false,false);
  auto model = std::make_shared<MuZero::ModelBundle>(directory);
  const auto precision = config.getBool("testFP16") ? enabled_t::True : enabled_t::False;
  const int inputsVersion = model->game() == "go" ? 7 : model->game() == "xiangqi" ? 9 : 8;
  auto batcher = std::make_unique<MuZero::Batcher>(model,8,std::vector<int>{0,0},precision,config);
  auto position = std::make_unique<MuZero::PositionEvaluator>(*batcher,false,true,0,8,4,&logger);
  auto* evaluator = position.get();
  SelfplayManager manager(2,&logger,1,true);
  auto settings = PlaySettings::loadForSelfplay(config,false);
  settings.recordTreePositions = true;
  settings.recordTreeThreshold = 2;
  settings.recordTreeTargetWeight = 0.125f;
  auto params = Setup::loadSingleParams(config,Setup::SETUP_FOR_OTHER);
  MatchPairer::BotSpec bot;
  bot.botIdx = 0;
  bot.botName = "muzero-play-test";
  bot.baseParams = params;
  bot.muZeroEval = evaluator;
  MakeDir::make(output);
  MakeDir::make(output + "/data");
  auto* writer = new TrainingDataWriter(output + "/data",inputsVersion,64,1.,
                            model->posLen(),model->posLen(),"muzero-play-data",true);
  manager.loadModelAndStartDataWriting(new SelfplayModel(bot.botName,directory,std::move(batcher),std::move(position)),writer,nullptr);
  SelfplayModel* acquired = manager.acquireLatest();
  testAssert(acquired && acquired->muZeroEvaluator() == evaluator && acquired->boardEvaluator() == nullptr);
  json report = json::array();
  for(int mode = 0; mode < 3; mode++) {
    bot.baseParams.useGumbel = mode > 0;
    bot.baseParams.useGumbelNonroot = mode == 2;
    bot.baseParams.gumbel.max_num_considered_actions = 4;
    GameRunner runner(config,"muzero-game-init",settings,logger);
    ForkData forks;
    for(int game = 0; game < 3; game++) {
      manager.countOneGameStarted(acquired);
      int roots = 0;
      auto onMove = [&](const Board& board, const BoardHistory& hist, Player pla, Loc move,
                        const std::vector<double>&, const Search* search) {
        testAssert(search->muZeroEvaluator == evaluator && search->nnEvaluator == nullptr);
        testAssert(hist.isLegal(board,move,pla));
        testAssert(search->getRootVisits() > 0);
        if(roots == 0) {
          std::vector<PolicyTargetMove> policy;
          std::vector<Loc> locs;
          std::vector<double> values;
          testAssert(Play::extractPolicyTarget(policy,search,search->rootNode,locs,values));
          Loc selected = Board::NULL_LOC;
          for(const auto& target: policy) if(target.policyTarget > 0) selected = target.loc;
          testAssert(selected != Board::NULL_LOC);
          SearchPolicyBuffer<bool> storage(search->getPolicyBufferSize(),true);
          bool* mask = storage.data();
          const int64_t visits = search->getRootVisits();
          double surprise, searchEntropy, policyEntropy;
          testAssert(!search->getPolicySurpriseAndEntropy(surprise,searchEntropy,policyEntropy,search->rootNode,mask));
          policy.clear();
          testAssert(!Play::extractPolicyTarget(policy,search,search->rootNode,locs,values,mask));
          testAssert(policy.empty());
          mask[search->getPos(selected)] = true;
          testAssert(Play::extractPolicyTarget(policy,search,search->rootNode,locs,values,mask));
          testAssert(policy.size() == 1 && policy[0].loc == selected && policy[0].policyTarget > 0);
          testAssert(search->getPolicySurpriseAndEntropy(surprise,searchEntropy,policyEntropy,search->rootNode,mask));
          testAssert(std::abs(surprise) < 1e-6 && std::abs(searchEntropy) < 1e-6 && std::abs(policyEntropy) < 1e-6);
          testAssert(search->getRootVisits() == visits);
        }
        roots++;
      };
      std::unique_ptr<FinishedGameData> data(runner.runGame("muzero-play-" + std::to_string(mode) + "-" + std::to_string(game),
        bot,bot,&forks,nullptr,logger,[]() noexcept {return false;},nullptr,nullptr,nullptr,onMove));
      testAssert(data && data->hasFullData && roots > 0);
      testAssert(data->changedNeuralNets.empty());
      testAssert(data->policyTargetsByTurn.size() == (size_t)roots);
      testAssert(data->whiteValueTargetsByTurn.size() == (size_t)roots + 1);
      Board board = data->startBoard;
      BoardHistory hist = data->startHist;
      Player pla = data->startPla;
      int reanalyzed = 0;
      for(int turn = 0; turn < roots; turn++) {
        const auto& policy = *data->policyTargetsByTurn[turn].policyTargets;
        int64_t mass = 0;
        for(const auto& target: policy) {
          testAssert(hist.isLegal(board,target.loc,pla));
          mass += target.policyTarget;
        }
        testAssert(mass > 0);
        const auto& reanalysis = data->reanalysisByTurn[turn];
        if(reanalysis.wasReanalyzed) {
          reanalyzed++;
          testAssert(!reanalysis.usedOutcomeTargets);
          testAssert(reanalysis.originalNumVisits <= 8 + params.numThreads);
        }
        const Move move = data->endHist.moveHistory[data->startHist.moveHistory.size() + turn];
        testAssert(move.pla == pla && hist.isLegal(board,move.loc,pla));
        hist.makeBoardMoveAssumeLegal(board,move.loc,pla,nullptr);
        pla = getOpp(pla);
      }
      int treeSides = 0;
      for(const auto* side: data->sidePositions) {
        testAssert(!side->hist.isGameFinished);
        for(const auto& target: side->policyTarget)
          testAssert(side->hist.isLegal(side->board,target.loc,side->pla));
        for(const auto& target: side->whiteQValueTargets.targets)
          testAssert(side->hist.isLegal(side->board,target.loc,side->pla));
        int mass = 0;
        for(const auto& target: side->policyTarget) mass += target.policyTarget;
        testAssert(mass > 0 && std::isfinite(side->policySurprise) && side->policySurprise >= 0.0);
        if(side->targetWeightUnrounded == settings.recordTreeTargetWeight) treeSides++;
        testAssert(side->playoutDoublingAdvantage == 0.0);
      }
      testAssert(reanalyzed == roots);
      report.push_back({{"mode",mode},{"game",game},{"roots",roots},{"reanalysis",reanalyzed},
                        {"side",data->sidePositions.size()},{"tree_side",treeSides},{"forks",forks.forks.size()},
                        {"used_fork",data->usedInitialPosition},{"opening",data->startHist.moveHistory.size()},
                        {"handicap",data->handicapForSgf},{"truncated",data->hitTurnLimit}});
      manager.enqueueDataToWrite(acquired,data.release());
    }
  }
  // The active old acquisition prevents cleanup when a newer service arrives.
  // Releasing it retires the service only after its source data queue is drained.
  MakeDir::make(output + "/next");
  auto nextBatcher = std::make_unique<MuZero::Batcher>(model,8,std::vector<int>{0},precision,config);
  auto nextPosition = std::make_unique<MuZero::PositionEvaluator>(*nextBatcher,false,false,0,4,2,&logger);
  auto* nextWriter = new TrainingDataWriter(output + "/next",inputsVersion,64,1.,
                                           model->posLen(),model->posLen(),"next-writer",true);
  manager.loadModelAndStartDataWriting(new SelfplayModel("next",directory,std::move(nextBatcher),std::move(nextPosition)),nextWriter,nullptr);
  testAssert(manager.numModels() == 2 && manager.hasModel(bot.botName));
  manager.release(acquired);
  testAssert(manager.numModels() == 1 && !manager.hasModel(bot.botName));
  testAssert(manager.getTotalNumRowsProcessed() > 0);
  auto* next = manager.acquireLatest();
  testAssert(next->getModelName() == "next");
  manager.release(next);
  std::ofstream out(output + "/report.json");
  out << report.dump(2);
}
