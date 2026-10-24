#include "etazero/go_position.h"
namespace etazero {
std::unique_ptr<RuleState> make_go_history_state(const ::BoardHistory&,int);
Game game_from_go_history(const ::BoardHistory& history,int canvas) {
    auto board=history.initialBoard;
    if(board.x_size>canvas||board.y_size>canvas)throw std::runtime_error("Go history exceeds model canvas");
    RuleOptions options;options.go_rules=history.rules.toJsonStringNoKomi();options.komi=history.rules.komi;options.go_pass_alive=history.modes.alwaysComputePassAliveUnderSuicideRules;options.go_exclude_atari=history.modes.excludeTerritoryAdjacentToAtari;
    Sgf::PositionSample initial;initial.board=board;initial.nextPla=history.initialPla;initial.initialTurnNumber=history.initialTurnNumber;initial.weight=1;initial.hintLoc=::Board::NULL_LOC;
    options.go_position=Sgf::PositionSample::toJsonLine(initial);
    std::vector<int> moves;for(const auto& move:history.moveHistory)moves.push_back(go_action(move.loc,board.x_size,canvas));
    return Game(std::max(board.x_size,board.y_size),canvas,Rule::GO,options,make_go_history_state(history,canvas),std::move(moves),history.getCurrentTurnNumber());
}
int go_action(Loc loc,int width,int canvas) {
    if(loc==::Board::NULL_LOC)return -1;
    if(loc==::Board::PASS_LOC)return canvas*canvas;
    return Location::getY(loc,width)*canvas+Location::getX(loc,width);
}
Game game_from_go_sample(const Sgf::PositionSample& sample,int canvas,const RuleOptions& original) {
    if(sample.board.x_size>canvas||sample.board.y_size>canvas)
        throw std::runtime_error("Go sample dimensions exceed model canvas");
    auto initial=sample;initial.moves.clear();auto options=original;
    options.go_position=Sgf::PositionSample::toJsonLine(initial);
    Game game(std::max(sample.board.x_size,sample.board.y_size),canvas,Rule::GO,options);
    for(const auto& move:sample.moves) {
        if(game.player()!=(move.pla==P_BLACK?1:-1))throw std::runtime_error("Nonalternating Go sample history");
        game.play(go_action(move.loc,sample.board.x_size,canvas));
    }
    return game;
}
}
