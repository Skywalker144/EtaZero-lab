#pragma once
#include "etazero/schema.h"
#include <stdexcept>
#include <vector>

namespace etazero {
// Requests use group indices: Gomoku D4 [0,7], Hex identity/180 [0,1].
// Hex White additionally transposes at the model boundary so own goal is vertical.
inline bool hex_input(const std::vector<float>& obs) {
    if (obs.size()<GLOBAL_FEATURES) throw std::runtime_error("Invalid observation shape");
    return obs[obs.size()-GLOBAL_FEATURES+HEX_GLOBAL]!=0;
}
inline int input_symmetry(const std::vector<float>& obs,int symmetry) {
    bool hex=hex_input(obs);
    if (symmetry<0 || symmetry>=(hex?2:8)) throw std::runtime_error("Symmetry outside game group");
    if (!hex) return symmetry;
    bool white=obs[obs.size()-GLOBAL_FEATURES+HEX_WHITE_GLOBAL]!=0;
    return (symmetry?2:0)+(white?4:0);
}
}
