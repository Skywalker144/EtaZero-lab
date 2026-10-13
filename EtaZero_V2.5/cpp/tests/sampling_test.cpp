#include "etazero/sampling.h"
#include <iostream>
#include <mutex>
using namespace etazero;
static void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
static void near(double a,double b) {check(std::abs(a-b)<1e-12,"Numerical sample mismatch");}
struct Conditioned : Evaluator {
    std::mutex mutex;std::vector<float> halfs,flags;
    Evaluation evaluate(const std::vector<float>& input) override {
        size_t n=(input.size()-GLOBAL_FEATURES)/INPUT_PLANES;size_t g=INPUT_PLANES*n;
        {std::lock_guard<std::mutex> lock(mutex);halfs.push_back(input[g+5]);flags.push_back(input[g+4]);}
        Evaluation e;e.logits.assign(n,0);e.wdl={.3+.05*input[g+5],.4,.3-.05*input[g+5]};return e;
    }
};
struct RankedFork : Evaluator {
    int calls=0;
    Evaluation evaluate(const std::vector<float>& input) override {
        ++calls;size_t n=(input.size()-GLOBAL_FEATURES)/INPUT_PLANES;
        Evaluation e;e.logits.assign(n,0);e.wdl=input[2*n]>0?WDL{.1,0,.9}:WDL{.5,0,.5};return e;
    }
};
struct ReplayBoundaryFork : Evaluator {
    Game before;
    int calls=0;
    bool nonterminal_candidate=false;
    explicit ReplayBoundaryFork(const Game& position):before(position) {}
    Evaluation evaluate(const std::vector<float>& input) override {
        ++calls;
        // A candidate must add exactly one stone to the known pre-final board.
        auto baseline=before.observation(-before.player());
        check(input.size()==baseline.size(),"Fork candidate observation shape");
        int action=-1,n=before.actions();
        for(size_t i=0;i<input.size();++i)if(input[i]!=baseline[i]) {
            check(action==-1 && i>=size_t(2*n) && i<size_t(3*n) && baseline[i]==0 && input[i]==1,
                  "Fork candidate preserves the exact pre-final board");
            action=static_cast<int>(i)-2*n;
        }
        check(before.legal(action),"Fork candidate adds a legal stone");
        Game next=before;next.play(action);
        check(input==next.observation(),"Fork candidate has the next-player perspective");
        nonterminal_candidate|=!next.finished();
        Evaluation e;e.logits.assign(n,0);
        e.wdl=next.finished()?WDL{.9,0,.1}:WDL{.1,0,.9};
        return e;
    }
};
static void check_hint_integrity() {
    auto directory=std::filesystem::temp_directory_path()/
        ("etazero_hint_test_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    auto positions=directory/"positions.txt",config=directory/"effective.cfg";
    auto write_config=[&](int probability,const std::string& hash) {
        std::ofstream output(config);output<<"[hint_positions]\nhint_positions_prob="<<probability
            <<"\npositions_file="<<positions.string()<<"\n";
        if(!hash.empty())output<<"positions_sha256="<<hash<<"\n";
    };
    // hashlib.sha256(b"5 freestyle 1 0 0\n"), independent of the native digest code.
    const std::string expected="3ea34288d64b676760e9bfb1e8d0fb419a346ed272f814cfa07c2a69bebdebcc";
    try {
        {std::ofstream output(positions);output<<"5 freestyle 1 0 0\n";}
        write_config(1,expected);Config enabled(config.string());
        auto loaded=load_hint_positions(enabled,6);
        check(loaded.size()==1 && loaded[0].hint_action==0,"Matching hint hash accepted");
        for(const char* content:{"5 freestyle 1 1 0\n","5 freestyle 1 0 0\r\n"}) {
            {std::ofstream output(positions,std::ios::binary);output<<content;}
            bool rejected=false;
            try{load_hint_positions(enabled,6);}catch(const std::runtime_error& error){rejected=std::string(error.what()).find("checksum mismatch")!=std::string::npos;}
            check(rejected,"Changed hint bytes rejected with unchanged configuration");
        }
        {std::ofstream output(positions);output<<"5 freestyle 1 0 0\n";}
        check(load_hint_positions(enabled,6).size()==1,"Restored hint bytes accepted");
        write_config(1,"");bool rejected=false;
        try{load_hint_positions(Config(config.string()),6);}catch(const std::runtime_error&){rejected=true;}
        check(rejected,"Enabled hints require recorded SHA256");
        write_config(0,"");std::filesystem::remove(positions);
        check(load_hint_positions(Config(config.string()),6).empty(),"Disabled hints do not require input/hash");
    } catch(...) {std::filesystem::remove_all(directory);throw;}
    std::filesystem::remove_all(directory);
}
int main(int argc,char** argv) {
    if(argc>1 && std::string(argv[1])=="--limits") {
        double d;int adv,pla,current,turn,hint,exact,hint_fork,force,cheap,count,clear;
        std::cout.precision(17);
        while(std::cin>>d>>adv>>pla>>current>>turn>>hint>>exact>>hint_fork>>force>>cheap>>clear>>count) {
            std::vector<double> history(count);for(auto& q:history)std::cin>>q;
            SelfplaySearchConfig c{400,70,.6,0,bool(clear),true,.9,3,350,.1};c.max_playouts=600;
            SelfplaySearchContext context{hint,turn,current,bool(exact),bool(hint_fork),bool(force)};
            auto limits=selfplay_search_limits(c,history,cheap,{d,adv},pla,context);
            std::cout<<limits.search.max_visits<<' '<<limits.search.max_playouts<<' '<<limits.target_weight<<' '<<limits.cheap_search<<' '
                     <<limits.search.clear_before_search<<' '<<limits.search.remove_root_noise<<' '<<limits.search.hint_action<<' '
                     <<cheap_search_probability(c,context)<<'\n';
        }return 0;
    }
    if(argc>1 && std::string(argv[1])=="--surprise") {
        int direct,size,winner,count;std::cout.precision(17);
        while(std::cin>>direct>>size>>winner>>count) {
            FinishedGame record{};record.size=size;record.winner=winner;
            for(int i=0;i<count;++i) {Step step{};std::cin>>step.player;for(auto& p:step.network_wdl)std::cin>>p;for(auto& p:step.search_wdl)std::cin>>p;record.steps.push_back(step);}
            compute_value_surprises(record,direct);for(const auto& step:record.steps)std::cout<<step.value_surprise<<' ';std::cout<<'\n';
        }return 0;
    }
    if(argc>1) {
        double d;int a,p,n,cheap;
        std::cout.precision(17);
        while(std::cin>>d>>a>>p>>n>>cheap) {
            SelfplaySearchConfig c{n,std::max(5,n/2),.5,0,true,false,.9,3,n,1};
            c.max_playouts=n;auto limits=selfplay_search_limits(c,{},cheap,{d,a},p);
            Game game(5,6,Rule::RENJU);game.set_pda(d,a);auto obs=game.observation(p);
            std::cout<<limits.search.max_visits<<' '<<limits.search.max_playouts<<' '<<limits.search.clear_before_search<<' '<<obs[5*36+4]<<' '<<obs[5*36+5]<<'\n';
        }
        return 0;
    }
    check_hint_integrity();
    SelfplaySearchConfig c{400,70,.6,0,false,true,.9,3,350,.1};
    auto strong=selfplay_search_limits(c,{},false,{3,1},1),weak=selfplay_search_limits(c,{},false,{3,1},-1);
    check(strong.search.max_visits==711 && weak.search.max_visits==89,"PDA full round budgets");
    check(strong.search.clear_before_search && weak.search.clear_before_search,"PDA clears reused roots");
    auto cheap=selfplay_search_limits(c,{},true,{3,1},-1);check(cheap.search.max_visits==16 && cheap.search.max_playouts==16 && cheap.search.clear_before_search && cheap.search.remove_root_noise,"PDA after PCR");
    auto reduced=selfplay_search_limits(c,{1,1,1},false,{3,1},1);check(reduced.search.max_visits==622 && reduced.target_weight<.100001,"PDA after Reduce Visits");
    c.max_playouts=600;auto explicit_cap=selfplay_search_limits(c,{},false,{3,1},1);check(explicit_cap.search.max_playouts==1067,"PDA explicit playout cap");
    auto too_small=c;too_small.max_playouts=20;bool cap_error=false;
    try{selfplay_search_limits(too_small,{},true);}catch(const std::runtime_error&){cap_error=true;}check(cap_error,"PCR rejects explicit ceiling below cheap visits");
    cap_error=false;try{selfplay_search_limits(too_small,{},false);}catch(const std::runtime_error&){cap_error=true;}check(cap_error,"Reduce rejects explicit ceiling below minimum");
    near(playout_budget_factor({3,1},1)+playout_budget_factor({3,1},-1),2);
    near(playout_budget_factor({3,1},1)/playout_budget_factor({3,1},-1),8);
    bool bad=false;try{selfplay_search_limits(c,{},true,{std::log2(100.0),1},-1);}catch(const std::runtime_error&){bad=true;}check(bad,"PDA does not silently clamp tiny caps");
    std::mt19937_64 rng(19);check(sample_playout_advantage(0,8,rng).doublings==0,"PDA disabled");
    bool both[2]={false,false};for(int i=0;i<200;++i) {auto a=sample_playout_advantage(1,8,rng);check(a.doublings>=0 && a.doublings<3,"PDA ratio uniform doubling range");both[a.player==1]=true;}check(both[0]&&both[1],"PDA samples both colors");
    Conditioned evaluator;Game game(5,6,Rule::RENJU);game.set_pda(3,1);
    auto b=game.observation(1),w=game.observation(-1);near(b[5*36+5],1.5);near(w[5*36+5],-1.5);check(b[5*36+4]==1 && w[5*36+4]==1,"PDA flag");
    AlphaZeroState a(game,evaluator);Game ordinary=game;ordinary.set_pda(0,0);AlphaZeroState z(ordinary,evaluator);check(a.graph_key()!=z.graph_key(),"PDA splits graph identity");
    SearchSettings settings{80,4,1,1,0,1,true};settings.graph_search=true;
    Search search(evaluator,settings,17);SearchRun limits;limits.max_visits=80;limits.clear_before_search=true;
    auto result=search.run(game,0,limits);check(result.root_visits==80 && result.initial_visits==0,"Conditioned search visits");
    check(std::find(evaluator.halfs.begin(),evaluator.halfs.end(),1.5)!=evaluator.halfs.end() && std::find(evaluator.halfs.begin(),evaluator.halfs.end(),-1.5)!=evaluator.halfs.end(),"PDA sign at real search depths");
    game.play(result.action);auto observation=game.observation();check(observation[5*36+5]==-1.5,"Game transition PDA sign");
    evaluator.halfs.clear();evaluator.flags.clear();FinishedGame record{};record.size=5;record.canvas=6;record.rule=Rule::RENJU;
    std::vector<Game> positions{game};size_t initial=positions.size();
    search_side_positions(positions,record,search,evaluator,30,true,0,[](const Game&){return 0;},rng,[]{return false;});
    check(!record.side_positions.empty() && record.side_positions.size()==positions.size(),"Side production queue");
    for(const auto& side:record.side_positions) {check(side.observation[5*36+4]==0 && side.observation[5*36+5]==0,"Side clears PDA");check(side.target_weight==1 && side.row_repeats==1,"Side source frequency");}
    for(float h:evaluator.halfs)check(h==0,"Side conditional inference cleared");
    for(float f:evaluator.flags)check(f==0,"Side conditional flag cleared");
    bool recursed=positions.size()>initial;
    for(int k=0;k<12 && !recursed;++k) {
        std::vector<Game> pending{game};FinishedGame sample{};
        search_side_positions(pending,sample,search,evaluator,10,false,0,[](const Game&){return 0;},rng,[]{return false;});recursed=pending.size()>1;
    }
    check(recursed,"Side recursive response plus fork path");
    std::vector<double> p(36,0);p[2]=1;
    for(int i=0;i<100;++i){int fork=sample_fork_move(ordinary,p,2,rng);check(fork==-1 || (fork!=2 && ordinary.legal(fork)),"Fork excludes actual action");}
    FinishedGame kl{};kl.size=5;kl.winner=1;Step surprise{};surprise.player=1;surprise.network_wdl={.5,0,.5};surprise.search_wdl={.9,0,.1};kl.steps.push_back(surprise);
    compute_value_surprises(kl,true);near(kl.steps[0].value_surprise,.9*std::log(1.8)+.1*std::log(.2));
    FinishedGame gate{};gate.size=5;gate.winner=1;
    for(int i=0;i<3;++i) {Step step{};step.player=i%2?-1:1;step.cheap_search=i==2;step.target_weight=i==2?0:1;step.policy_surprise=i==2?20:1;step.network_wdl=step.search_wdl={.5,0,.5};gate.steps.push_back(step);}
    FinishedGame excess=gate;apply_training_weights(excess,.5,0,rng,true,false);check(excess.steps[2].target_weight>0,"Ordinary cheap excess recovers rows");
    apply_training_weights(gate,.5,0,rng,true,true);near(gate.steps[0].target_weight,1);near(gate.steps[1].target_weight,1);near(gate.steps[2].target_weight,0);
    FinishedGame trajectory{};trajectory.size=5;trajectory.canvas=6;trajectory.rule=Rule::RENJU;
    Game real(5,6,Rule::RENJU);real.set_pda(1,1);std::vector<Game> originals;std::vector<double> history;
    for(int action:{0,6,1,7,2,8,3,9,4}) {
        originals.push_back(real);history.push_back(.95);Step step{};step.player=real.player();step.action=action;step.observation=real.observation();step.visits.assign(36,0);step.visits[action]=3;step.policy.assign(36,0);step.policy[action]=1;step.policy_target=quantize_policy(step.policy);step.cheap_search=real.turn()%2==0;step.target_weight=step.cheap_search?0:1;step.policy_surprise=real.turn()+1;
        real.play(action);trajectory.steps.push_back(step);
    }
    trajectory.winner=real.winner();ReanalysisConfig reanalysis;reanalysis.enabled=true;reanalysis.proportion=1;reanalysis.direct_value_surprise=true;reanalysis.use_outcome_targets=false;
    auto original=trajectory;SelfplaySearchConfig small{40,20,.6,0,true,true,.9,3,30,.1};
    reanalyze_positions(trajectory,originals,history,small,{1,1},reanalysis,search,[](const Game&){return 0;},rng,[]{return false;});
    for(size_t i=0;i<trajectory.steps.size();++i) {
        const auto& step=trajectory.steps[i];check(step.action==original.steps[i].action,"Reanalysis preserves played action");
        check(step.reanalyzed==step.cheap_search,"Cheap-only no-replacement selection");
        if(step.reanalyzed) {check(step.reanalysis_original_visits==4 && step.reanalysis_policy_surprise==i+1,"Original reanalysis diagnostics");check(!step.reanalysis_used_outcome,"Explicit outcome gate");check(step.target_weight>0 && step.policy_target.size()==36,"Full reanalysis target replacement");}
    }
    FinishedGame no_surprise{};for(int i=0;i<3;++i){Step step{};step.cheap_search=true;step.policy_surprise=step.value_surprise=0;no_surprise.steps.push_back(step);}
    auto all_zero=select_reanalysis_turns(no_surprise,reanalysis,rng);check(all_zero==std::vector<size_t>({0,1,2}),"All-zero reanalysis uniform fallback and no replacement");
    ReanalysisConfig zero;zero.enabled=true;std::mt19937_64 zero_rng(9),control_rng(9);
    check(select_reanalysis_turns(trajectory,zero,zero_rng).empty() && zero_rng()==control_rng(),"Zero reanalysis probability preserves RNG");
    ReanalysisConfig disabled;check(select_reanalysis_turns(trajectory,disabled,rng).empty(),"Default no reanalysis");
    SelfplaySearchContext hint{12,2,2,true,false,false};c.max_playouts=600;
    auto hinted=selfplay_search_limits(c,{1,1,1},true,{3,1},1,hint);
    check(!hinted.cheap_search && hinted.search.hint_action==12 && hinted.search.max_visits==2844 && hinted.search.max_playouts==4267 && hinted.target_weight==1 && !hinted.search.remove_root_noise,"Exact hint fourfold full before PDA; no PCR/reduce");
    near(cheap_search_probability(c,hint),0);hint.exact_hint=false;hint.current_turn=7;near(cheap_search_probability(c,hint),.3);
    hint.current_turn=8;near(cheap_search_probability(c,hint),.6);hint.hint_action=-1;hint.hint_fork=true;hint.current_turn=2;near(cheap_search_probability(c,hint),.3);
    hint.force_full=true;near(cheap_search_probability(c,hint),0);
    auto forced=selfplay_search_limits(c,{1,1,1},true,{3,1},1,hint);check(!forced.cheap_search && forced.search.hint_action==-1 && forced.search.max_visits==622,"Reanalysis force full skips hint/PCR, retains reduced and PDA");
    Game empty(5,6,Rule::RENJU);auto context=hint_context(empty,empty,12,InitialKind::Hint);check(context.exact_hint,"Hint position match");
    Game changed=empty;changed.play(0);check(!hint_context(changed,empty,12,InitialKind::Hint).exact_hint,"Moved position is not exact hint");
    SearchSettings guided{120,1,1,1,0,1,true};Search hint_search(evaluator,guided,3);SearchRun guided_run;guided_run.max_visits=120;guided_run.hint_action=12;
    auto guided_result=hint_search.run(empty,0,guided_run);
    float ordinary_prior=(1.f/25)*(1.f-.02f);float expected_hint=ordinary_prior+float(25*((1.f/25)*.02f));
    check(std::abs(guided_result.search_policy[12]-expected_hint)<1e-7 && std::abs(guided_result.search_policy[0]-ordinary_prior)<1e-7,"Source float hint prior transfer");
    int64_t max_other=0;for(int a=0;a<36;++a)if(a!=12)max_other=std::max(max_other,guided_result.visits[a]);
    check(guided_result.visits[12]+2>=.8*max_other,"Root hint receives near-most-visited exploration");
    guided_run.hint_action=-1;auto neutral=hint_search.run(empty,0,guided_run);check(neutral.initial_visits==0,"Hint removal clears corrupted tree");
    guided_run.hint_action=5;bool illegal_hint=false;try{hint_search.run(empty,0,guided_run);}catch(const std::runtime_error&){illegal_hint=true;}check(illegal_hint,"Hint padding rejected");
    GameForkConfig early_config(1,1,0,3,12,36),late_config(0,1,.025,3,12,36);
    auto early=make_game_fork(original,early_config,evaluator,false,0,rng,[]{return false;});
    check(early && early->kind==InitialKind::EarlyFork && early->actions.size()==1 && !early->game.finished() && early->game.pda_doublings()==0,"Early fork exponential zero-start and early priority");
    FinishedGame short_game{};short_game.size=short_game.canvas=15;short_game.rule=Rule::FREESTYLE;
    for(int a:{0,15,1,16,2,17,3,18,4}) {Step s{};s.action=a;short_game.steps.push_back(s);}
    Game pre_final(15,15,Rule::FREESTYLE);
    for(int a:{0,15,1,16,2,17,3,18})pre_final.play(a);
    check(pre_final.turn()==8 && !pre_final.finished(),"Known fork boundary precedes the winning move");
    GameForkConfig tail_config(1,0,.025,3,12,36);
    bool pre_final_seen=false,equal_seen=false,beyond_seen=false;
    for(int seed=1;seed<=256;++seed) {
        std::mt19937_64 predict(seed),actual(seed);
        std::bernoulli_distribution(1)(predict);
        size_t index=std::floor(std::exponential_distribution<double>(1)(predict)*.025*225);
        if(index<short_game.steps.size()-1)continue;
        pre_final_seen|=index==short_game.steps.size()-1;
        equal_seen|=index==short_game.steps.size();beyond_seen|=index>short_game.steps.size();
        ReplayBoundaryFork tail_evaluator(pre_final);
        auto fork=make_game_fork(short_game,tail_config,tail_evaluator,false,0,actual,[]{return false;});
        check(tail_evaluator.calls>=3 && tail_evaluator.calls<=12,"Early exponential tail evaluates candidates before terminal move");
        check(bool(fork)==tail_evaluator.nonterminal_candidate,"Early tail retains a selected nonterminal candidate");
        if(fork) {
            check(fork->kind==InitialKind::EarlyFork && fork->actions.size()==9 && fork->game.turn()==9 && !fork->game.finished(),
                  "Early tail records the eight-move prefix and candidate");
            for(size_t i=0;i<8;++i)check(fork->actions[i]==short_game.steps[i].action,"Early tail preserves the recorded prefix");
        }
    }
    check(pre_final_seen && equal_seen && beyond_seen,"Early tail covers pre-final, equal and oversized indices");
    RankedFork ranked;FinishedGame empty_record{};empty_record.size=5;empty_record.canvas=6;empty_record.rule=Rule::FREESTYLE;
    GameForkConfig fixed_choices(1,0,0,36,36,36);std::mt19937_64 fork_rng(772);
    int best_count=0;constexpr int trials=2048;
    for(int i=0;i<trials;++i) {
        auto fork=make_game_fork(empty_record,fixed_choices,ranked,false,0,fork_rng,[]{return false;});
        check(fork && fork->actions.size()==1,"Ranked fork is nonterminal");best_count+=fork->actions[0]==0;
    }
    check(ranked.calls==36*trials,"Source candidate count retained above legal-action count");
    double best_probability=1-std::pow(24.0/25,36);
    check(std::abs(double(best_count)/trials-best_probability)<.04,"Fork best-move probability follows independent with-replacement formula");
    bool late_seen=false;for(int i=0;i<20 && !late_seen;++i){auto f=make_game_fork(original,late_config,evaluator,false,0,rng,[]{return false;});late_seen=f && f->kind==InitialKind::GameFork && !f->game.finished();}check(late_seen,"Late fork replay and finite candidate set");
    ForkPool pool;pool.add(*early);auto pulled=pool.take(rng);check(pulled && pulled->actions==early->actions && !pool.take(rng),"Shared fork pool consumes each position once");
    FinishedGame hinted_record=original;hinted_record.opening.initial_position_kind=1;hinted_record.opening.hint_action=12;
    auto hint_fork=make_hint_fork(hinted_record);check(hint_fork && hint_fork->kind==InitialKind::HintFork && hint_fork->actions==std::vector<int>{12},"Unchosen hint creates actual hint continuation");
    hinted_record.opening.hint_action=0;check(!make_hint_fork(hinted_record),"Chosen hint does not fork");
    hinted_record.opening.hint_action=12;hinted_record.steps[0].search_wdl={.9,0,.1};hinted_record.steps[1].search_wdl={.2,.3,.5};
    compute_value_surprises(hinted_record,true);check(hinted_record.steps[0].search_wdl==WDL{.5,.3,.2},"First hinted target copies next value in correct perspective");
    hinted_record.steps.resize(1);hinted_record.winner=1;compute_value_surprises(hinted_record,true);check(hinted_record.steps[0].search_wdl==WDL{1,0,0},"Single hinted turn copies true terminal value");
    std::cout<<"PDA, side, hint and fork checks passed\n";
}
