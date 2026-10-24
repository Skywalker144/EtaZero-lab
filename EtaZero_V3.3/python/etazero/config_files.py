"""KataGo key/value syntax and in-place includes, with EtaZero section namespaces."""
from pathlib import Path
import re
import json

# Only spelling changes are listed here. Section namespaces disambiguate the
# independent learner, selfplay and analysis settings; they do not change values.
SOURCE_KEYS = {
    'reportAnalysisWinratesAs':'report_analysis_winrates_as','analysisPVLen':'analysis_pv_len','alwaysComputePassAliveUnderSuicideRules':'always_compute_pass_alive_under_suicide_rules','excludeTerritoryAdjacentToAtari':'exclude_territory_adjacent_to_atari',
    'analysisWideRootNoise':'analysis_wide_root_noise','analysisIgnorePreRootHistory':'analysis_ignore_pre_root_history',
    'defaultBoardXSize':'board_width','defaultBoardYSize':'board_height',
    'rootDirichletNoiseWeight':'noise_fraction','rootDirichletNoiseTotalConcentration':'dirichlet_total_concentration',
    'numAnalysisThreads': 'num_analysis_threads',
    'maxVisits': 'visits', 'numSearchThreads': 'search_threads',
    'cpuctExploration': 'c_puct', 'cpuctExplorationLog': 'c_puct_log',
    'cpuctExplorationBase': 'c_puct_base', 'cpuctUtilityStdevPrior': 'c_puct_stdev_prior',
    'cpuctUtilityStdevPriorWeight': 'c_puct_stdev_prior_weight', 'cpuctUtilityStdevScale': 'c_puct_stdev_scale',
    'numVirtualLossesPerThread': 'virtual_loss', 'useLcbForSelection': 'use_lcb',
    'minVisitPropForLCB': 'min_visit_prop_for_lcb',
    'fpuParentWeightByVisitedPolicyPow': 'fpu_parent_weight_by_visited_policy_pow',
    'fpuParentWeight': 'fpu_parent_weight',
    'chosenMoveTemperature': 'temperature', 'chosenMoveTemperatureEarly': 'temperature_early',
    'chosenMoveTemperatureHalflife': 'temperature_halflife',
    'chosenMoveTemperatureOnlyBelowProb': 'temperature_only_below_prob',
    'nnMaxBatchSize': 'max_batch', 'numNNServerThreadsPerModel': 'server_threads',
    'nnRandomize': 'nn_randomize', 'nnRandomizeSymmetries': 'nn_randomize',
    'nnCacheSize': 'cache_entries', 'nnMutexPoolSizePowerOfTwo': 'nn_mutex_pool_power_of_two',
    'subtreeValueBiasTableNumShards': 'subtree_value_bias_table_num_shards',
    'playoutDoublingAdvantagePla': 'playout_doubling_advantage_player',
    'visitCapContemptPla': 'visit_cap_contempt_player',
    'rules': 'go_rules', 'boardSize': 'board_size',
    'startPosesFromSgfDir': 'files', 'startPosesSgfExcludes': 'excludes',
    'startPosesProb': 'probability', 'startPosesLoadProb': 'load_probability',
    'startPosesTurnWeightLambda': 'turn_weight_lambda',
    'hintPosesDir': 'positions_file', 'hintPosesProb': 'hint_positions_prob',
    'humanSLProfile': 'human_sl_profile',
    'humanSLRootExploreProbWeightless': 'human_sl_root_weightless',
    'humanSLRootExploreProbWeightful': 'human_sl_root_weightful',
    'humanSLPlaExploreProbWeightless': 'human_sl_player_weightless',
    'humanSLPlaExploreProbWeightful': 'human_sl_player_weightful',
    'humanSLOppExploreProbWeightless': 'human_sl_opponent_weightless',
    'humanSLOppExploreProbWeightful': 'human_sl_opponent_weightful',
    'humanSLChosenMoveProp': 'human_sl_chosen_move_proportion',
}


def _snake(key):
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2',
                  re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', key)).lower()


def _value(raw, location, *, allow_empty=False):
    raw = raw.lstrip()
    if not raw.startswith('"'):
        value = raw.split('#', 1)[0].strip()
    else:
        value, i = '', 1
        while i < len(raw) and raw[i] != '"':
            if raw[i] == '\\':
                i += 1
                if i == len(raw):
                    raise ValueError(f'Unterminated escape: {location}')
            value += raw[i]
            i += 1
        if i == len(raw) or (raw[i+1:].strip() and not raw[i+1:].lstrip().startswith('#')):
            raise ValueError(f'Invalid quoted value: {location}')
    if not value and not allow_empty:
        raise ValueError(f'Empty configuration value: {location}')
    return value


def read_config_file(path, fields, *, local_only=(), section_aliases=None, ancestors=(),
                     keys_override=False, includes_override=True):
    """Read in source order; duplicate local keys are errors unless enabled.

    Flat common keys use KataGo names. Sections and empty section values are
    EtaZero extensions needed for separate game/learner profiles. Unsupported
    fields fail at this public boundary rather than silently disabling a feature.
    """
    section_aliases = section_aliases or {}
    path = Path(path).resolve()
    if path in ancestors:
        raise ValueError('Configuration include cycle: ' + ' -> '.join(map(str, (*ancestors, path))))
    values, seen, section = {}, set(), None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        location = f'{path}:{number}'
        if not line or line.startswith(('#', ';')):
            continue
        if line.startswith('@'):
            match = re.fullmatch(r'@include[\s=]+(.+)', line)
            if not match:
                raise ValueError(f'Invalid include: {location}')
            filename = _value(match[1],location).strip("'")
            included = read_config_file(path.parent / filename, fields, local_only=local_only,
                section_aliases=section_aliases, ancestors=(*ancestors, path),
                keys_override=keys_override, includes_override=includes_override)
            for key in local_only:
                included.pop(key, None)
            if not includes_override and values.keys() & included.keys():
                raise ValueError(f'Duplicate configuration key in include: {location}')
            for target,value in included.items():
                if target in values and target[1] in ('pattern_corpora','auto_avoid_repeat_sizes'):
                    previous=json.loads(values[target]);incoming=json.loads(value)
                    if isinstance(previous,list):
                        while len(previous)<len(incoming):previous.append({})
                        for i,entry in enumerate(incoming):previous[i].update(entry)
                    else:
                        for size,entry in incoming.items():previous.setdefault(size,{}).update(entry)
                    value=json.dumps(previous,separators=(',',':'))
                values[target]=value
            continue
        if line.startswith('['):
            match = re.fullmatch(r'\[([^]]+)\]\s*(?:#.*)?', line)
            if not match:
                raise ValueError(f'Invalid section: {location}')
            section = match[1]
            if section == 'DEFAULT':
                raise ValueError(f'DEFAULT fields are not supported: {location}')
            if section not in fields:
                raise ValueError(f'Wrong section/file ownership: {location}: [{section}]')
            continue
        match = re.fullmatch(r'([a-zA-Z0-9_-]+)\s*=(.*)', line)
        if not match:
            raise ValueError(f'Invalid key/value pair: {location}')
        name, raw = match.groups()
        # Numbered SGF corpora and board-size overrides share the source flat spelling.
        corpus = re.fullmatch(r'avoidSgf(\d*)(PatternUtility|PatternLambda|PatternMinTurnNumber|PatternMaxFiles|PatternAllowedNames|PatternDirs)',name)
        sized = re.fullmatch(r'autoAvoidRepeat(Utility|Lambda|MinTurnNumber|MaxTurnNumber|MaxPoses|SaveChunkSize)(\d+x\d+)',name)
        if corpus or sized:
            group=section or ('engine' if 'engine' in fields else 'patterns')
            target=(section_aliases.get(group,group),'pattern_corpora' if corpus else 'auto_avoid_repeat_sizes')
            value=_value(raw,location)
            if corpus:
                records=json.loads(values.get(target,'[]'));index=int(corpus[1] or '1')-1
                if not 0<=index<99999:raise ValueError(f'Invalid SGF corpus number: {location}')
                while len(records)<=index:records.append({})
                prop={'PatternUtility':'penalty','PatternLambda':'decay','PatternMinTurnNumber':'min_turn',
                      'PatternMaxFiles':'max_files','PatternAllowedNames':'players','PatternDirs':'paths'}[corpus[2]]
                records[index][prop]=[v.strip() for v in value.split(',') if v.strip()] if prop in ('players','paths') else float(value) if prop in ('penalty','decay') else int(value)
            else:
                records=json.loads(values.get(target,'{}'));prop=_snake('autoAvoidRepeat'+sized[1]);records.setdefault(sized[2],{})[prop]=float(value) if sized[1] in ('Utility','Lambda') else int(value)
            if name in seen and not keys_override:raise ValueError(f'Duplicate configuration key: {location}: {name}')
            seen.add(name);values[target]=json.dumps(records,separators=(',',':'));continue
        key = name if section is not None and name in fields[section] else SOURCE_KEYS.get(name, _snake(name))
        if name=='maxVisits' and 'search' in fields and 'full_search_visits' in fields['search']:
            key='full_search_visits'
        if name=='rules' and section is None and 'go' in fields:
            key='rules'
        if section is None:
            candidates = [group for group, keys in fields.items() if key in keys]
            if name.startswith('startPoses'):
                candidates = [g for g in candidates if g == 'start_positions']
            if name.startswith('hintPoses'):
                candidates = [g for g in candidates if g == 'hint_positions']
            if 'engine' in candidates:
                group = 'engine'
            elif len(candidates) == 1:
                group = candidates[0]
            else:
                raise ValueError(f'Unknown key or ambiguous namespace: {location}: {name}')
        else:
            group = section
            if key not in fields[group]:
                raise ValueError(f'Unknown key: {location}: {group}.{name}')
        local_key = (group, key)
        target = (section_aliases.get(group, group), key)
        if local_key in seen and not keys_override:
            raise ValueError(f'Duplicate configuration key: {location}: {name}')
        if target in values and not includes_override:
            raise ValueError(f'Duplicate configuration key across files: {location}: {name}')
        values[target] = _value(raw, location, allow_empty=section is not None)
        seen.add(local_key)
    return values
