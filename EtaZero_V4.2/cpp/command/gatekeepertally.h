#ifndef COMMAND_GATEKEEPERTALLY_H_
#define COMMAND_GATEKEEPERTALLY_H_

#include "../game/boardhistory.h"

struct GatekeeperTally {
  int games = 0;
  int wins = 0;
  int losses = 0;
  int draws = 0;
  int noResults = 0;
  double candidatePoints = 0.0;
  double baselinePoints = 0.0;

  void record(const BoardHistory& history, Player candidate, double noResultUtilityForWhite) {
    double points;
    if(history.isNoResult) {
      noResults++;
      const double whitePoints = 0.5 * (1.0 + noResultUtilityForWhite);
      points = candidate == P_WHITE ? whitePoints : 1.0 - whitePoints;
    }
    else if(history.winner == C_EMPTY) {
      draws++;
      points = 0.5;
    }
    else if(history.winner == candidate) {
      wins++;
      points = 1.0;
    }
    else {
      losses++;
      points = 0.0;
    }
    games++;
    candidatePoints += points;
    baselinePoints += 1.0 - points;
  }
};

#endif
