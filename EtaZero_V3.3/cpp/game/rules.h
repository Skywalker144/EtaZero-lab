#pragma once
#include "etazero/schema.h"
#include "game/geometry.h"
#include <algorithm>
#include <array>
#include <string>
#include <stdexcept>
#include <vector>

namespace etazero {
struct Board : BoardGeometry {
    std::vector<int> cells;
    explicit Board(int edge) : Board(edge,edge,edge) {}
    Board(int width,int height,int stride) : BoardGeometry{width,height,stride} {
        if(width<2 || height<2 || width>25 || height>25 || stride<width || stride>25)
            throw std::runtime_error("Invalid board geometry");
        cells.assign(storage_size(),0);
    }
};
Rule parse_rule(const std::string& name);
}
