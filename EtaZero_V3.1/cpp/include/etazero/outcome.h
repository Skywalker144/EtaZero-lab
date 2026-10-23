#pragma once
#include "etazero/schema.h"
#include <array>
namespace etazero {
// An outcome is read only after finished(). Winner uses absolute player color.
struct Outcome {
    int winner=0;
    Reason reason=Reason::DRAW;
    bool has_result() const { return reason!=Reason::NO_RESULT; }
    double value(int player) const { return winner*player; }
    double draw_probability() const { return winner==0 && has_result()?1:0; }
    std::array<double,3> wdl(int player) const {
        if(!has_result())return {.5,0,.5};
        return {double(winner==player),double(winner==0),double(winner==-player)};
    }
};
}
