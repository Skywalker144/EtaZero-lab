#pragma once
#include "rules.h"
namespace etazero {
class Game {
    Board board_;
    int canvas_, player_ = 1, turn_ = 0, winner_ = 0, reason_ = 0;
    Rule rule_;
    bool finished_ = false;
public:
    explicit Game(int size, int canvas, Rule rule) : board_(size), canvas_(canvas), rule_(rule) {
        if (canvas < size || canvas > 25) throw std::runtime_error("Invalid canvas size");
    }
    int size() const { return board_.size; }
    int canvas() const { return canvas_; }
    int actions() const { return canvas_ * canvas_; }
    int player() const { return player_; }
    int turn() const { return turn_; }
    int winner() const { return winner_; }
    int reason() const { return reason_; }
    Rule rule() const { return rule_; }
    bool finished() const { return finished_; }
    const Board& board() const { return board_; }
    bool legal(int a) const {
        return !finished_ && a >= 0 && a < actions() && a / canvas_ < size() && a % canvas_ < size() &&
               board_.cells[a / canvas_ * size() + a % canvas_] == 0;
    }
    std::vector<float> observation() const {
        std::vector<float> out(INPUT_PLANES * actions(), 0);
        for (int y = 0; y < size(); ++y) for (int x = 0; x < size(); ++x) {
            int a = y * canvas_ + x, cell = board_.cells[y * size() + x];
            out[a] = cell == player_;
            out[actions() + a] = cell == -player_;
            out[2 * actions() + a] = player_ == 1;
            out[3 * actions() + a] = 1;
            out[4 * actions() + a] = rule_ == Rule::STANDARD;
            out[5 * actions() + a] = rule_ == Rule::RENJU;
        }
        return out;
    }
    double terminal_value() const {
        if (!finished_) throw std::runtime_error("State is not terminal");
        return winner_ * player_;
    }
    void play(int a) {
        if (!legal(a)) throw std::runtime_error("Illegal move: " + std::to_string(a));
        int local = a / canvas_ * size() + a % canvas_;
        if (rule_ == Rule::RENJU && player_ == 1) {
            RenjuAnalyzer analyzer;
            if (analyzer.forbidden(board_, local)) { winner_ = -1; reason_ = 2; }
        }
        board_.cells[local] = player_;
        ++turn_;
        if (!winner_) for (int n : board_.lengths(local, player_))
            if (n == 5 || (n > 5 && (rule_ == Rule::FREESTYLE || (rule_ == Rule::RENJU && player_ == -1)))) {
                winner_ = player_; reason_ = 1;
            }
        finished_ = winner_ || turn_ == size() * size();
        player_ = -player_;
    }
};
}
