#include "game/rules.h"
namespace etazero {
Rule parse_rule(const std::string& name) {
    for (int i = 0; i < static_cast<int>(std::size(RULE_NAMES)); ++i)
        if (name == RULE_NAMES[i]) return static_cast<Rule>(i);
    throw std::runtime_error("Unknown game rule: " + name);
}
}
