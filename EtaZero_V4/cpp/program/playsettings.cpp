#include "../program/playsettings.h"

#include "../program/setup.h"

PlaySettings::PlaySettings()
  :balancedOpening(),initGamesWithPolicy(false),policyInitAreaProp(0.0),startPosesPolicyInitAreaProp(0.0),
   policyInitGammaShape(1.0),policyInitAvgMoveNum(0.0),startPosesPolicyInitAvgMoveNum(0.0),sidePositionProb(0.0),
   policyInitAreaTemperature(1.0),handicapTemperature(1.0),
   earlyForkGameProb(0.0),earlyForkGameExpectedMoveProp(0.0),forkGameProb(0.0),forkGameMinChoices(1),earlyForkGameMaxChoices(1),forkGameMaxChoices(1),
   cheapSearchProb(0),cheapSearchVisits(0),cheapSearchTargetWeight(0.0f),
   reduceVisits(false),reduceVisitsThreshold(100.0),reduceVisitsThresholdLookback(1),reducedVisitsMin(0),reducedVisitsWeight(1.0f),
   policySurpriseDataWeight(0.0),valueSurpriseDataWeight(0.0),useSearchValueSurprise(false),scaleDataWeight(1.0),
   useReanalyze(false),reanalyzeProp(0.0),
   reanalyzePolicySurpriseWeight(1.0),reanalyzeValueSurpriseWeight(1.0),reanalyzeSurpriseExponent(1.0),
   reanalyzeUseOutcomeTargets(true),
   recordTreePositions(false),recordTreeThreshold(0),recordTreeTargetWeight(0.0f),
   noResolveTargetWeights(false),
   allowResignation(false),resignThreshold(0.0),resignConsecTurns(1),
   forSelfPlay(false),
   handicapAsymmetricPlayoutProb(0.0),normalAsymmetricPlayoutProb(0.0),maxAsymmetricRatio(2.0),
   recordTimePerMove(false),
   logSearchTreeOptions(Setup::defaultLogSearchTreeOptions())
{}
PlaySettings::~PlaySettings()
{}

static void rejectKomiCompensation(ConfigParser& cfg) {
  for(const char* key : {"estimateLeadVisits", "estimateLeadProb"}) {
    if(cfg.contains(key))
      throw StringError(std::string("Score supervision is excluded from EtaZero; removed setting: ") + key);
  }
  for(const char* key : {"compensateAfterPolicyInitProb", "compensateKomiVisits", "flipKomiProbWhenNoCompensate", "fancyKomiVarying", "minAsymmetricCompensateKomiProb", "dynamicSelfKomiBonusMin", "dynamicSelfKomiBonusMax", "dynamicSelfKomiWinLossMin", "dynamicSelfKomiWinLossMax"}) {
    if(cfg.contains(key))
      throw StringError(std::string("EtaZero uses fixed rule komi; removed setting: ") + key);
  }
}

static void loadPolicyInit(PlaySettings& settings, ConfigParser& cfg, bool selfplay) {
  const auto game = cfg.contains("game") ? Rules::parseGame(cfg.getString("game")) : Rules::Game::GO;
  const bool gomoku = Rules::forGame(game).isPlacement() && game != Rules::Game::HEX;
  if(gomoku) {
    settings.policyInitAvgMoveNum = selfplay && !cfg.contains("policyInitAvgMoveNum") ? 12.0 :
      cfg.getDouble("policyInitAvgMoveNum",0.0,100.0);
    settings.startPosesPolicyInitAvgMoveNum = cfg.contains("startPosesPolicyInitAvgMoveNum") ?
      cfg.getDouble("startPosesPolicyInitAvgMoveNum",0.0,100.0) : 0.0;
  }
  else {
    settings.policyInitAreaProp = selfplay && !cfg.contains("policyInitAreaProp") ? 0.04 :
      cfg.getDouble("policyInitAreaProp",0.0,1.0);
    settings.startPosesPolicyInitAreaProp = cfg.contains("startPosesPolicyInitAreaProp") ?
      cfg.getDouble("startPosesPolicyInitAreaProp",0.0,1.0) : 0.0;
    settings.policyInitGammaShape = cfg.contains("policyInitGammaShape") ?
      cfg.getDouble("policyInitGammaShape",0.5,selfplay ? 10.0 : 100.0) : 1.0;
  }
  settings.policyInitAreaTemperature = cfg.contains("policyInitAreaTemperature") ?
    cfg.getDouble("policyInitAreaTemperature",0.1,5.0) : 1.0;
}

PlaySettings PlaySettings::loadForMatch(ConfigParser& cfg) {
  rejectKomiCompensation(cfg);
  if(cfg.contains("sekiForkHackProb"))
    throw StringError("sekiForkHackProb is excluded from EtaZero");
  PlaySettings playSettings;
  playSettings.balancedOpening = BalancedOpening::Settings::load(cfg,false);
  playSettings.allowResignation = cfg.getBool("allowResignation");
  playSettings.resignThreshold = cfg.getDouble("resignThreshold",-1.0,0.0); //Threshold on [-1,1], regardless of winLossUtilityFactor
  playSettings.resignConsecTurns = cfg.getInt("resignConsecTurns",1,100);
  playSettings.initGamesWithPolicy =  cfg.contains("initGamesWithPolicy") ? cfg.getBool("initGamesWithPolicy") : false;
  if(playSettings.initGamesWithPolicy) {
    loadPolicyInit(playSettings,cfg,false);
  }

  playSettings.recordTimePerMove = true;
  playSettings.logSearchTreeOptions = Setup::loadLogSearchTreeOptions(cfg);
  return playSettings;
}

PlaySettings PlaySettings::loadForGatekeeper(ConfigParser& cfg) {
  rejectKomiCompensation(cfg);
  if(cfg.contains("sekiForkHackProb"))
    throw StringError("sekiForkHackProb is excluded from EtaZero");
  PlaySettings playSettings;
  playSettings.balancedOpening = BalancedOpening::Settings::load(cfg,false);
  playSettings.allowResignation = cfg.getBool("allowResignation");
  playSettings.resignThreshold = cfg.getDouble("resignThreshold",-1.0,0.0); //Threshold on [-1,1], regardless of winLossUtilityFactor
  playSettings.resignConsecTurns = cfg.getInt("resignConsecTurns",1,100);
  playSettings.logSearchTreeOptions = Setup::loadLogSearchTreeOptions(cfg);
  return playSettings;
}

PlaySettings PlaySettings::loadForSelfplay(ConfigParser& cfg, bool /*isDistributed*/) {
  rejectKomiCompensation(cfg);
  if(cfg.contains("sekiForkHackProb"))
    throw StringError("sekiForkHackProb is excluded from EtaZero");
  PlaySettings playSettings;
  playSettings.balancedOpening = BalancedOpening::Settings::load(cfg,true);
  playSettings.initGamesWithPolicy = cfg.getBool("initGamesWithPolicy");
  loadPolicyInit(playSettings,cfg,true);
  playSettings.sidePositionProb =
    //forkSidePositionProb is the legacy name, included for backward compatibility
    (cfg.contains("forkSidePositionProb") && !cfg.contains("sidePositionProb")) ?
    cfg.getDouble("forkSidePositionProb",0.0,1.0) : cfg.getDouble("sidePositionProb",0.0,1.0);

  playSettings.handicapTemperature = cfg.contains("handicapTemperature") ? cfg.getDouble("handicapTemperature",0.1,5.0) : 1.0;


  playSettings.earlyForkGameProb = cfg.getDouble("earlyForkGameProb",0.0,0.5);
  playSettings.earlyForkGameExpectedMoveProp = cfg.getDouble("earlyForkGameExpectedMoveProp",0.0,1.0);
  playSettings.forkGameProb = cfg.getDouble("forkGameProb",0,0.5);
  playSettings.forkGameMinChoices = cfg.getInt("forkGameMinChoices",1,100);
  playSettings.earlyForkGameMaxChoices = cfg.getInt("earlyForkGameMaxChoices",1,100);
  playSettings.forkGameMaxChoices = cfg.getInt("forkGameMaxChoices",1,100);
  playSettings.cheapSearchProb = cfg.getDouble("cheapSearchProb",0.0,1.0);
  playSettings.cheapSearchVisits = cfg.getInt("cheapSearchVisits",1,10000000);
  playSettings.cheapSearchTargetWeight = cfg.getFloat("cheapSearchTargetWeight",0.0f,1.0f);
  playSettings.reduceVisits = cfg.getBool("reduceVisits");
  playSettings.reduceVisitsThreshold = cfg.getDouble("reduceVisitsThreshold",0.0,0.999999);
  playSettings.reduceVisitsThresholdLookback = cfg.getInt("reduceVisitsThresholdLookback",0,1000);
  playSettings.reducedVisitsMin = cfg.getInt("reducedVisitsMin",1,10000000);
  playSettings.reducedVisitsWeight = cfg.getFloat("reducedVisitsWeight",0.0f,1.0f);
  playSettings.policySurpriseDataWeight = cfg.getDouble("policySurpriseDataWeight",0.0,1.0);
  playSettings.valueSurpriseDataWeight = cfg.getDouble("valueSurpriseDataWeight",0.0,1.0);
  playSettings.useSearchValueSurprise = cfg.contains("useSearchValueSurprise") ? cfg.getBool("useSearchValueSurprise") : false;
  playSettings.scaleDataWeight = cfg.contains("scaleDataWeight") ? cfg.getDouble("scaleDataWeight",0.01,10.0) : 1.0;
  playSettings.useReanalyze = cfg.contains("useReanalyze") ? cfg.getBool("useReanalyze") : false;
  if(playSettings.useReanalyze) {
    playSettings.reanalyzeProp = cfg.getDouble("reanalyzeProp",0.0,1.0);
    playSettings.reanalyzePolicySurpriseWeight = cfg.getDouble("reanalyzePolicySurpriseWeight",0.0,100.0);
    playSettings.reanalyzeValueSurpriseWeight = cfg.getDouble("reanalyzeValueSurpriseWeight",0.0,100.0);
    playSettings.reanalyzeSurpriseExponent = cfg.getDouble("reanalyzeSurpriseExponent",0.0,10.0);
    playSettings.reanalyzeUseOutcomeTargets = cfg.getBool("reanalyzeUseOutcomeTargets");
  }
  playSettings.handicapAsymmetricPlayoutProb = cfg.getDouble("handicapAsymmetricPlayoutProb",0.0,1.0);
  playSettings.normalAsymmetricPlayoutProb = cfg.getDouble("normalAsymmetricPlayoutProb",0.0,1.0);
  playSettings.maxAsymmetricRatio = cfg.getDouble("maxAsymmetricRatio",1.0,100.0);
  playSettings.forSelfPlay = true;
  playSettings.logSearchTreeOptions = Setup::loadLogSearchTreeOptions(cfg);

  if(playSettings.policySurpriseDataWeight + playSettings.valueSurpriseDataWeight > 1.0)
    throw StringError("policySurpriseDataWeight + valueSurpriseDataWeight > 1.0");

  return playSettings;
}
