// Offline trajectory audit uses the same complete rule state as selfplay.
#include "etazero/game.h"
#include <nlohmann/json.hpp>
#include <iostream>
using namespace etazero;
using Json=nlohmann::json;
int main() {
    try {
        std::string line;
        while(std::getline(std::cin,line)) {
            auto q=Json::parse(line); RuleOptions options;
            auto o=q.value("options",Json::object());options.go_rules=o.value("go_rules","chinese");
            options.komi=o.value("komi",7.5);options.fen=o.value("fen","");
            Game game(q.at("size"),q.at("canvas"),parse_rule(q.at("rule")),options);
            game.set_pda(q.value("pda",0.),q.value("pda_player",0));
            auto actions=q.at("actions").get<std::vector<int>>();
            for(size_t i=0;i<=actions.size();++i) {
                std::vector<int> legal;for(int a=0;a<game.actions();++a)if(game.legal(a))legal.push_back(a);
                Json row={{"observation",game.observation()},{"player",game.player()},{"legal",legal},
                    {"finished",game.finished()},{"winner",game.winner()},{"reason",int(game.reason())},{"fen",game.fen()}};
                std::cout<<row.dump()<<'\n';
                if(i<actions.size())game.play(actions[i]);
            }
        }
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
