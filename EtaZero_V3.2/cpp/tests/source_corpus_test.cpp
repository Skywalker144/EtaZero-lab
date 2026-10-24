#include "etazero/go_position.h"
#include "etazero/sampling.h"
#include "etazero/pattern_bonus.h"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace etazero;
static void check(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main() {
    auto dir=std::filesystem::temp_directory_path()/std::filesystem::path("etazero_source_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    try {
        Game init(5,6,Rule::GO);std::mt19937_64 rng(7);
        auto sgf=dir/"branches.sgf",cfg=dir/"config.cfg",exclude=dir/"exclude.txt";
        const std::string content="(;FF[4]GM[1]SZ[5]AB[aa][ee]AW[bb]PL[W]PB[black]PW[white];W[cc](;B[dd])(;B[de]))";
        {std::ofstream f(sgf);f<<content;}
        auto config=[&](bool exclusions) {
            std::ofstream f(cfg);f<<"[environment]\nrules=go\n[start_positions]\nprobability=1\nload_probability=1\nturn_weight_lambda=0\nfiles="<<sgf.string()<<','<<sgf.string()<<"\nexcludes="<<(exclusions?exclude.string():"")<<"\n[hint_positions]\nhint_positions_prob=1\npositions_file="<<(dir/"sample.hintposes.txt").string()<<"\n"
                <<"[analysis]\nauto_avoid_repeat_dir="<<(dir/"auto").string()<<"\nauto_avoid_repeat_sizes={}\nauto_avoid_repeat_utility=.2\nauto_avoid_repeat_lambda=.5\nauto_avoid_repeat_min_turn_number=0\nauto_avoid_repeat_max_turn_number=1000000\nauto_avoid_repeat_max_poses=1\nauto_avoid_repeat_save_chunk_size=1\n";
        };
        config(false);auto starts=load_start_positions(Config(cfg.string()),6,rng);
        check(starts.size()==4,"Setup plus both branches deduplicated across input files");
        bool setup=false,left=false,right=false;
        for(const auto& p:starts) {
            check(p.game.board().cells[0]==1&&p.game.board().cells[6]==-1,"SGF setup stones survive conversion");
            setup|=p.game.moves().empty()&&p.game.player()==-1;
            left|=p.game.board().cells[18]==1;right|=p.game.board().cells[23]==1;
        }
        check(setup&&left&&right,"SGF side to move and variations");
        std::unique_ptr<Sgf> parsed(Sgf::loadFile(sgf.string()));
        {std::ofstream f(exclude);f<<Global::uint64ToHexString(parsed->hash.hash0)<<Global::uint64ToHexString(parsed->hash.hash1)<<" # source hash exclusion\n";}
        config(true);check(load_start_positions(Config(cfg.string()),6,rng).empty(),"SGF hash exclusion with comment");config(false);
        Sgf::PositionSample sample;sample.board=::Board(5,5);sample.board.setStone(Location::getLoc(0,0,5),P_BLACK);
        sample.nextPla=P_WHITE;sample.initialTurnNumber=42;sample.weight=2;sample.trainingWeight=.25;
        sample.hintLoc=Location::getLoc(2,2,5);
        auto line=Sgf::PositionSample::toJsonLine(sample);
        {std::ofstream f(dir/"sample.hintposes.txt");f<<line<<'\n';}
        auto hints=load_hint_positions(Config(cfg.string()),6);
        check(hints.size()==1&&hints[0].game.turn()==42&&hints[0].game.player()==-1&&hints[0].hint_action==14,"PositionSample turn and hint");
        check(hints[0].sampling_weight==2&&hints[0].training_weight==.25,"Independent sampling and loss weights");
        auto game=game_from_go_sample(sample,6,{});
        check(PatternBonusTable::save_auto({line,line},Config(cfg.string()),"analysis"),"Save automatic history");
        auto table=PatternBonusTable::load_auto(Config(cfg.string()),"analysis",6,{});
        check(table&&std::abs(table->get(game.pattern_key(14,0,false))+.2)<1e-12,"Automatic patterns use white sign and per-file deduplication");
        // The first entire file is loaded even when maxPoses is crossed within it.
        auto old=dir/"auto/5x5/old_poses.txt";{std::ofstream f(old);f<<line<<'\n';}
        std::filesystem::last_write_time(old,std::filesystem::file_time_type::clock::now()-std::chrono::hours(1));
        PatternBonusTable::load_auto(Config(cfg.string()),"analysis",6,{});
        check(!std::filesystem::exists(old),"Automatic history prunes old files at file boundaries");
        sample.hintLoc=::Board::NULL_LOC;{std::ofstream f(dir/"sample.hintposes.txt");f<<Sgf::PositionSample::toJsonLine(sample)<<'\n';}
        check(load_hint_positions(Config(cfg.string()),6).front().hint_action==-1,"Optional source hint move");
        for(auto dimensions:{std::pair{3,5},std::pair{5,3}}) {
            auto rectangle=sample;rectangle.board=::Board(dimensions.first,dimensions.second);
            auto from_sample=game_from_go_sample(rectangle,6,{});
            auto from_history=game_from_go_history(from_sample.go_history(),6);
            for(const auto* position:{&from_sample,&from_history}) {
                check(position->options().go_width==dimensions.first&&position->options().go_height==dimensions.second,
                      "Source sample and history retain actual dimensions in record options");
                Game restored(5,6,Rule::GO,position->options());
                check(restored.width()==dimensions.first&&restored.height()==dimensions.second,"Rectangular record options replay geometry");
            }
        }
        std::filesystem::remove_all(dir);std::cout<<"source corpus passed\n";
    }catch(...){std::filesystem::remove_all(dir);throw;}
}
