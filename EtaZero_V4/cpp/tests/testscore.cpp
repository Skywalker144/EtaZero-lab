#include "../tests/tests.h"

#include "../neuralnet/nninputs.h"

using namespace std;
using namespace TestCommon;

void Tests::runScoreTests() {
  cout << "Running rules score tests" << endl;
  ostringstream out;

  auto printScoreStats = [&out](const Board& /*board*/, const BoardHistory& hist) {
    out << "Black self komi wins/draw=0.5: " << hist.currentSelfKomi(P_BLACK, 0.5) << endl;
    out << "White self komi wins/draw=0.5: " << hist.currentSelfKomi(P_WHITE, 0.5) << endl;
    out << "Black self komi wins/draw=0.25: " << hist.currentSelfKomi(P_BLACK, 0.25) << endl;
    out << "White self komi wins/draw=0.25: " << hist.currentSelfKomi(P_WHITE, 0.25) << endl;
    out << "Black self komi wins/draw=0.75: " << hist.currentSelfKomi(P_BLACK, 0.75) << endl;
    out << "White self komi wins/draw=0.75: " << hist.currentSelfKomi(P_WHITE, 0.75) << endl;

    out << "Winner: " << PlayerIO::colorToChar(hist.winner) << endl;
    double score = hist.finalWhiteMinusBlackScore;
    out << "Final score: " << score << endl;

  };

  {
    const char* name = "On-board even 9x9, komi 7.5";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "On-board even 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
.........
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }

  {
    const char* name = "On-board black ahead 7 9x9, komi 7";

    Board board = Board::parseBoard(9,9,R"%%(
.........
.........
ooooooooo
.........
.........
xxxxxxx..
xxxxxxxxx
.........
.........
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);

    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }


  {
    const char* name = "On-board even 5x5, komi 7";

    Board board = Board::parseBoard(5,5,R"%%(
.....
ooooo
.....
xxxxx
.....
)%%");

    Rules rules = Rules::getTrompTaylorish();
    rules.komi = 7.0;
    BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes(false,false));
    hist.endAndScoreGameNow(board);

    printScoreStats(board,hist);
    cout << name << endl;
    cout << out.str() << endl;
    cout << endl;
  }



}
