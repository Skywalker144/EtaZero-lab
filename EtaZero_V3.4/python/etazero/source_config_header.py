"""Generate native public search spellings from the Python configuration boundary."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from etazero.config_files import SOURCE_KEYS
from etazero.engine_config import ENGINE_FIELDS
from etazero.config import SEARCH_PARAMETERS, SOURCE_SEARCH_BOUNDS, boolean
fields=set(SEARCH_PARAMETERS)|set('visits search_threads c_puct virtual_loss reuse_tree temperature temperature_early use_fpu fpu_reduction_max root_fpu_reduction_max fpu_parent_weight_by_visited_policy_pow use_lcb lcb_stdevs min_visit_prop_for_lcb policy_target_pruning'.split())
fields|={'playout_doubling_advantage','playout_doubling_advantage_player','root_noise_enabled','noise_fraction','dirichlet_total_concentration','shaped_dirichlet_noise','root_desired_per_child_visits_coeff'}
fields-=set('human_model human_sl_profile pattern_corpora'.split())
fields={k for k in fields if not k.startswith('auto_avoid_repeat')}
aliases={k:k for k in fields}
for key in fields:
    words=key.split('_');aliases[words[0]+''.join(w.capitalize() for w in words[1:])]=key
aliases.update({k:v for k,v in SOURCE_KEYS.items() if v in fields})
lines=['#pragma once','#include <map>','#include <string>','namespace etazero {',
       'inline const std::map<std::string,std::string> SOURCE_SEARCH_KEYS = {']
lines.extend('    {"'+k+'","'+v+'"},' for k,v in sorted(aliases.items()))
lines.extend(['};','inline const std::map<std::string,char> SOURCE_SEARCH_TYPES = {'])
for key in sorted(fields):
    kind='i' if ENGINE_FIELDS[key] is int else 'f' if ENGINE_FIELDS[key] is float else 'b' if ENGINE_FIELDS[key] is boolean else 's'
    lines.append('    {"'+key+'",\''+kind+'\'},')
lines.extend(['};','inline const std::map<std::string,std::pair<double,double>> SOURCE_SEARCH_RANGES = {'])
for key,(low,high) in sorted(SOURCE_SEARCH_BOUNDS.items()):
    if key in fields:lines.append('    {"'+key+'",{'+str(float(low))+','+str(float(high))+'}},')
lines.extend(['};','}'])
Path(sys.argv[1]).write_text('\n'.join(lines)+'\n')
