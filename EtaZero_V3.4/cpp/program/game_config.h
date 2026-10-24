#pragma once
#include "core/config.h"
#include "game/rule_state.h"
namespace etazero {
enum class GameConfigMode { SELFPLAY, ANALYSIS, MATCH };
// Configuration belongs to the caller; RuleState only receives typed values.
inline RuleOptions rule_options(const Config& c,GameConfigMode mode) {
    RuleOptions options;
    if(c.contains("go.rules"))options.go_rules=c.text("go.rules");
    if(c.contains("go.komi"))options.komi=c.number("go.komi");
    const std::string prefix=mode==GameConfigMode::ANALYSIS?"analysis.":"match.";
    if(mode!=GameConfigMode::SELFPLAY) {
        if(c.contains(prefix+"go_rules"))options.go_rules=c.text(prefix+"go_rules");
        if(c.contains(prefix+"komi"))options.komi=c.number(prefix+"komi");
    }
    return options;
}
}
