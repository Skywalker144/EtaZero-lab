#pragma once
#include <cstdint>

namespace etazero {
// Opaque to search: game adapters define the pattern and sharing conditions.
// Equality denotes a shared residual estimate, not identical game states.
// Placement games/Go pack exact patterns; Xiangqi hashes its larger pattern to 128 bits.
struct SubtreeValueBiasKey {
    uint64_t pattern=0, moves=0;
    bool operator<(const SubtreeValueBiasKey& other) const {
        return pattern<other.pattern || (pattern==other.pattern && moves<other.moves);
    }
    bool operator==(const SubtreeValueBiasKey& other) const { return pattern==other.pattern && moves==other.moves; }
};
}
