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
    std::vector<float> observation() const { return observation(player_); }
    // Counterfactual side-to-move input used by balanced opening root rejection.
    std::vector<float> observation(int perspective, bool forbidden_feature = true) const {
        if (perspective != 1 && perspective != -1) throw std::runtime_error("Invalid observation perspective");
        const int spatial = INPUT_PLANES * actions();
        std::vector<float> out(spatial + GLOBAL_FEATURES, 0);
        RenjuAnalyzer analyzer;
        const bool enabled = rule_ == Rule::RENJU && forbidden_feature;
        for (int y = 0; y < size(); ++y) for (int x = 0; x < size(); ++x) {
            int a = y * canvas_ + x, local = y * size() + x, cell = board_.cells[local];
            out[a] = 1;
            out[actions() + a] = cell == perspective;
            out[2 * actions() + a] = cell == -perspective;
            if (enabled && !cell && analyzer.forbidden(board_, local))
                out[(perspective == 1 ? 3 : 4) * actions() + a] = 1;
        }
        out[spatial] = rule_ == Rule::STANDARD;
        out[spatial + 1] = rule_ == Rule::RENJU;
        out[spatial + 2] = rule_ == Rule::RENJU ? -perspective : 0;
        out[spatial + 3] = enabled;
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
