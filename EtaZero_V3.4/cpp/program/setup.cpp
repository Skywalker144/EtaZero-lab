#include "search/muzero/gumbel_search.h"
#include "program/setup.h"
#include "search/search.h"
#include "search/gumbel_search.h"
#include "search/muzero/search.h"
namespace etazero {
SearchSettings settings(const Config& c, const std::string& mode) {
    bool selfplay = mode == "selfplay";
    std::string section = selfplay ? "search" : mode == "match" ? "match" : "analysis";
    auto key = [&](const std::string& group, const std::string& name) {
        return (selfplay ? group : section) + "." + name;
    };
    SearchSettings s{selfplay?c.integer64("search.full_search_visits")-1:std::max<int64_t>(0,c.integer64(section+".visits")-1),
                     c.integer(key("parallelism","search_threads")),
                     c.number(key("puct","c_puct")),c.number(key("puct","virtual_loss")),
                     selfplay?c.number("dirichlet_noise.noise_fraction"):c.contains(section+".root_noise_enabled")&&c.boolean(section+".root_noise_enabled")?c.number(section+".noise_fraction"):0,
                     selfplay?c.number("dirichlet_noise.dirichlet_total_concentration"):c.contains(section+".dirichlet_total_concentration")?c.number(section+".dirichlet_total_concentration"):1,c.boolean(key("search","reuse_tree")),
                     selfplay?0:c.integer64(section+".visits")};
    if(c.contains(key("patterns","avoid_repeated_pattern_utility")))s.avoid_repeated_pattern_utility=c.number(key("patterns","avoid_repeated_pattern_utility"));
    if(c.contains(key("patterns","pattern_corpora")))s.external_patterns=PatternBonusTable::load(c.text(key("patterns","pattern_corpora")),c.integer("network.canvas"),RuleOptions{});
    s.use_eval_cache=c.contains(key("graph_search","use_eval_cache"))&&c.boolean(key("graph_search","use_eval_cache"));
    s.eval_cache_min_visits=c.contains(key("graph_search","eval_cache_min_visits"))?c.integer64(key("graph_search","eval_cache_min_visits")):100;
    s.graph_search=c.boolean(key("graph_search","use_graph_search"));
    if(c.contains(key("graph_search","graph_search_rep_bound")))s.graph_rep_bound=c.integer(key("graph_search","graph_search_rep_bound"));
    s.graph_catch_up_leak_prob=c.number(key("graph_search","graph_search_catch_up_leak_prob"));
    if(c.contains(key("graph_search","node_table_shards_power_of_two")))s.node_table_shards_power_of_two=c.integer(key("graph_search","node_table_shards_power_of_two"));
    s.min_playouts_per_thread=c.contains(key("parallelism","min_playouts_per_thread"))?c.number(key("parallelism","min_playouts_per_thread")):(section=="analysis"?8:0);
    s.ignore_pre_root_history=c.contains(key("search_history","ignore_pre_root_history"))&&c.boolean(key("search_history","ignore_pre_root_history"));
    s.ignore_all_history=c.contains(key("search_history","ignore_all_history"))&&c.boolean(key("search_history","ignore_all_history"));
    if(c.contains(key("time_control","max_visits_pondering")))s.max_visits_pondering=c.integer64(key("time_control","max_visits_pondering"));
    if(c.contains(key("time_control","max_playouts_pondering")))s.max_playouts_pondering=c.integer64(key("time_control","max_playouts_pondering"));
    if(c.contains(key("time_control","max_time_pondering")))s.max_time_pondering=c.number(key("time_control","max_time_pondering"));
    if(c.contains(key("time_control","lag_buffer")))s.lag_buffer=c.number(key("time_control","lag_buffer"));
    if(c.contains(key("time_control","tree_reuse_carry_over_time_factor")))s.tree_reuse_carry_over_time_factor=c.number(key("time_control","tree_reuse_carry_over_time_factor"));
    if(c.contains(key("time_control","overallocate_time_factor")))s.overallocate_time_factor=c.number(key("time_control","overallocate_time_factor"));
    if(c.contains(key("time_control","midgame_time_factor")))s.midgame_time_factor=c.number(key("time_control","midgame_time_factor"));
    if(c.contains(key("time_control","midgame_turn_peak_time")))s.midgame_turn_peak_time=c.number(key("time_control","midgame_turn_peak_time"));
    if(c.contains(key("time_control","endgame_turn_time_decay")))s.endgame_turn_time_decay=c.number(key("time_control","endgame_turn_time_decay"));
    if(c.contains(key("time_control","obvious_moves_time_factor")))s.obvious_moves_time_factor=c.number(key("time_control","obvious_moves_time_factor"));
    if(c.contains(key("time_control","obvious_moves_policy_entropy_tolerance")))s.obvious_moves_policy_entropy_tolerance=c.number(key("time_control","obvious_moves_policy_entropy_tolerance"));
    if(c.contains(key("time_control","obvious_moves_policy_surprise_tolerance")))s.obvious_moves_policy_surprise_tolerance=c.number(key("time_control","obvious_moves_policy_surprise_tolerance"));
    if(c.contains(key("human_sl","human_sl_cpuct_exploration")))s.human_sl_cpuct_exploration=c.number(key("human_sl","human_sl_cpuct_exploration"));
    if(c.contains(key("human_sl","human_sl_cpuct_permanent")))s.human_sl_cpuct_permanent=c.number(key("human_sl","human_sl_cpuct_permanent"));
    if(c.contains(key("human_sl","human_sl_root_weightless")))s.human_sl_root_weightless=c.number(key("human_sl","human_sl_root_weightless"));
    if(c.contains(key("human_sl","human_sl_root_weightful")))s.human_sl_root_weightful=c.number(key("human_sl","human_sl_root_weightful"));
    if(c.contains(key("human_sl","human_sl_player_weightless")))s.human_sl_player_weightless=c.number(key("human_sl","human_sl_player_weightless"));
    if(c.contains(key("human_sl","human_sl_player_weightful")))s.human_sl_player_weightful=c.number(key("human_sl","human_sl_player_weightful"));
    if(c.contains(key("human_sl","human_sl_opponent_weightless")))s.human_sl_opponent_weightless=c.number(key("human_sl","human_sl_opponent_weightless"));
    if(c.contains(key("human_sl","human_sl_opponent_weightful")))s.human_sl_opponent_weightful=c.number(key("human_sl","human_sl_opponent_weightful"));
    if(c.contains(key("human_sl","human_sl_chosen_move_proportion")))s.human_sl_chosen_move_proportion=c.number(key("human_sl","human_sl_chosen_move_proportion"));
    if(c.contains(key("human_sl","human_sl_chosen_move_pikl_lambda")))s.human_sl_chosen_move_pikl_lambda=c.number(key("human_sl","human_sl_chosen_move_pikl_lambda"));
    if(c.contains(key("human_sl","human_sl_chosen_move_ignore_pass")))s.human_sl_chosen_move_ignore_pass=c.boolean(key("human_sl","human_sl_chosen_move_ignore_pass"));
    s.root_symmetry_pruning=c.contains(key("search_extras","root_symmetry_pruning"))?c.boolean(key("search_extras","root_symmetry_pruning")):(section=="analysis");
    s.wide_root_noise=c.contains(key("search_extras","wide_root_noise"))?c.number(key("search_extras","wide_root_noise")):(section=="analysis"?.04:0);
    s.futile_visits_threshold=c.contains(key("search_extras","futile_visits_threshold"))?c.number(key("search_extras","futile_visits_threshold")):0;
    s.visit_cap_contempt=c.contains(key("search_extras","visit_cap_contempt"))?c.integer64(key("search_extras","visit_cap_contempt")):0;
    s.visit_cap_contempt_player=c.contains(key("search_extras","visit_cap_contempt_player"))?c.integer(key("search_extras","visit_cap_contempt_player")):0;
    s.use_uncertainty=c.boolean(key("uncertainty","use_uncertainty"));
    s.uncertainty_coeff=c.number(key("uncertainty","uncertainty_coeff"));
    s.uncertainty_exponent=c.number(key("uncertainty","uncertainty_exponent"));
    s.uncertainty_max_weight=c.number(key("uncertainty","uncertainty_max_weight"));
    s.policy_optimism=c.number(key("optimistic_policy","policy_optimism"));
    s.root_policy_optimism=c.number(key("optimistic_policy","root_policy_optimism"));
    s.use_noise_pruning=c.boolean(key("noise_pruning","use_noise_pruning"));
    s.noise_prune_utility_scale=c.number(key("noise_pruning","noise_prune_utility_scale"));
    s.noise_pruning_cap=c.number(key("noise_pruning","noise_pruning_cap"));
    s.subtree_value_bias_factor=c.number(key("subtree_value_bias","subtree_value_bias_factor"));
    s.subtree_value_bias_free_prop=c.number(key("subtree_value_bias","subtree_value_bias_free_prop"));
    s.subtree_value_bias_weight_exponent=c.number(key("subtree_value_bias","subtree_value_bias_weight_exponent"));
    s.subtree_value_bias_table_num_shards=c.integer(key("subtree_value_bias","subtree_value_bias_table_num_shards"));
    s.use_fpu=c.boolean(key("fpu","use_fpu"));s.fpu_reduction_max=c.number(key("fpu","fpu_reduction_max"));
    s.root_fpu_reduction_max=c.number(key("fpu","root_fpu_reduction_max"));
    s.fpu_parent_power=c.number(key("fpu","fpu_parent_weight_by_visited_policy_pow"));
    s.fpu_parent_weight_by_visited_policy=c.boolean(key("fpu","fpu_parent_weight_by_visited_policy"));
    s.fpu_parent_weight=c.number(key("fpu","fpu_parent_weight"));
    s.max_playouts=c.integer64(key("search","max_playouts"));s.max_time=c.number(key("search","max_time"));
    s.use_lcb=c.boolean(key("lcb","use_lcb"));s.lcb_stdevs=c.number(key("lcb","lcb_stdevs"));
    s.use_non_buggy_lcb=c.contains(key("lcb","use_non_buggy_lcb"))?c.boolean(key("lcb","use_non_buggy_lcb")):!selfplay;
    s.min_lcb_visit_prop=c.number(key("lcb","min_visit_prop_for_lcb"));
    s.policy_target_pruning=c.boolean(key("policy_target","policy_target_pruning"));
    s.value_weight_exponent=c.number(key("value_weighting","value_weight_exponent"));
    s.chosen_move_subtract=c.number(key("policy_target","chosen_move_subtract"));s.chosen_move_prune=c.number(key("policy_target","chosen_move_prune"));
    s.fpu_loss_prop=c.number(key("fpu","fpu_loss_prop"));s.root_fpu_loss_prop=c.number(key("fpu","root_fpu_loss_prop"));
    s.c_puct_log=c.number(key("puct","c_puct_log"));s.c_puct_base=c.number(key("puct","c_puct_base"));
    s.c_puct_stdev_prior=c.number(key("puct","c_puct_stdev_prior"));s.c_puct_stdev_prior_weight=c.number(key("puct","c_puct_stdev_prior_weight"));
    s.c_puct_stdev_scale=c.number(key("puct","c_puct_stdev_scale"));
    s.root_symmetries=c.integer(key("symmetry","root_num_symmetries_to_sample"));
    s.nn_randomize=c.boolean(key("symmetry","nn_randomize"));s.nn_symmetry=c.integer(key("symmetry","nn_symmetry"));
    s.nn_policy_temperature=c.number(key("temperature","nn_policy_temperature"));
    s.root_policy_temperature=c.number(key("temperature","root_policy_temperature"));
    s.root_policy_temperature_early=c.number(key("temperature","root_policy_temperature_early"));
    s.temperature_halflife=c.number(key("temperature","temperature_halflife"));
    s.chosen_move_temperature_only_below_prob=c.number(key("temperature","temperature_only_below_prob"));
    if(selfplay){s.shaped_noise=c.boolean("dirichlet_noise.shaped_dirichlet_noise");s.forced_playouts=c.number("forced_playouts.root_desired_per_child_visits_coeff");}
    if(!selfplay&&c.contains(section+".root_noise_enabled")){s.shaped_noise=c.boolean(section+".shaped_dirichlet_noise");s.forced_playouts=c.number(section+".root_desired_per_child_visits_coeff");}
    return s;
}
double move_temperature(const Config& c,const std::string& mode,const Game& game) {
    bool selfplay=mode=="selfplay";
    std::string section=selfplay?"temperature":mode=="match"?"match":"analysis";
    return temperature_at_turn(c.number(section+(selfplay?".temperature":".temperature_early")),
        c.number(section+(selfplay?".final_temperature":".temperature")),
        c.number(section+".temperature_halflife"),game.turn(),game.area());
}
std::unique_ptr<GameSearch> search_for(InferenceService& evaluator,SearchSettings settings,uint64_t seed,const Config& c,const std::string& mode) {
    std::string prefix=mode=="selfplay"?"agent":mode;
    std::string root=c.contains(prefix+".root_search_algo")?c.text(prefix+".root_search_algo"):"puct";
    std::string nonroot=c.contains(prefix+".nonroot_search_algo")?c.text(prefix+".nonroot_search_algo"):"puct";
    if((root!="puct"&&root!="gumbel")||(nonroot!="puct"&&nonroot!="gumbel")||(root=="puct"&&nonroot!="puct"))
        throw std::runtime_error("Invalid root/nonroot search combination");
    if(root=="gumbel") {
        GumbelSettings g;auto key=[&](const std::string& name){return mode=="selfplay"?"gumbel."+name:mode+".gumbel_"+name;};
        if(c.contains(key("max_num_considered_actions")))g.max_num_considered_actions=c.integer(key("max_num_considered_actions"));
        if(c.contains(key("c_visit")))g.c_visit=c.number(key("c_visit"));
        if(c.contains(key("c_scale")))g.c_scale=c.number(key("c_scale"));
        g.noise_scale=mode=="selfplay"?1:0;
        if(c.contains(key("noise_scale")))g.noise_scale=c.number(key("noise_scale"));
        if(c.contains(key("rescale_q_values")))g.rescale_q_values=c.boolean(key("rescale_q_values"));
        if(c.contains(key("action_selection"))) {
            auto behavior=c.text(key("action_selection"));
            if(behavior!="gumbel"&&behavior!="visit")throw std::runtime_error("Unknown Gumbel action selection");
            g.sample_visits=behavior=="visit";
        }
        if(auto mu=dynamic_cast<muzero::Evaluator*>(&evaluator)) {
            if(nonroot=="gumbel")return std::make_unique<muzero::GumbelSearch<true>>(*mu,settings,g,seed);
            return std::make_unique<muzero::GumbelSearch<false>>(*mu,settings,g,seed);
        }
        if(nonroot=="gumbel")return std::make_unique<GumbelSearch<true>>(evaluator,settings,g,seed);
        return std::make_unique<GumbelSearch<false>>(evaluator,settings,g,seed);
    }
    if(auto mu=dynamic_cast<muzero::Evaluator*>(&evaluator))return std::make_unique<muzero::Search>(*mu,settings,seed);
    return std::make_unique<Search>(evaluator,settings,seed);
}
}
