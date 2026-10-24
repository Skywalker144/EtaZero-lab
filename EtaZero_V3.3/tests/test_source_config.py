import json
import pytest
from etazero.config_files import read_config_file
from etazero.engine_config import load_engine_config


def test_source_quotes_include_order_and_duplicate_keys(tmp_path):
    parent=tmp_path/'a#b.cfg';parent.write_text('x = "a # b\\\"c"\ny=first\n')
    child=tmp_path/'child.cfg';child.write_text('y=before\n@include "a#b.cfg" # inside quotes\nz="100%"\n')
    fields={'engine':dict.fromkeys(['x','y','z'])}
    values=read_config_file(child,fields)
    assert values=={('engine','x'):'a # b"c',('engine','y'):'first',('engine','z'):'100%'}
    child.write_text('x=one\nx=two\n')
    with pytest.raises(ValueError,match='Duplicate'):read_config_file(child,fields)


def test_source_minimal_analysis_defaults_and_spelling(tmp_path):
    path=tmp_path/'source.cfg'
    path.write_text('numSearchThreads=2\nmaxVisits=32\nfpuParentWeightByVisitedPolicy=TRUE\n'
                    'fpuParentWeightByVisitedPolicyPow=1.5\nrootPolicyTemperature=1.3\n')
    c=load_engine_config(path,environ={})['analysis']
    assert c['visits']==32 and c['search_threads']==2 and c['rule']=='go'
    assert c['fpu_parent_weight_by_visited_policy'] and c['fpu_parent_weight_by_visited_policy_pow']==1.5
    assert c['root_policy_temperature_early']==1.3 and c['c_puct_log']==.45
    assert c['min_playouts_per_thread']==8 and c['root_symmetry_pruning']
    path.write_text(path.read_text()+'nnPolicyTemperature=5.01\n')
    with pytest.raises(ValueError):load_engine_config(path,environ={})


def test_numbered_pattern_sources_merge_across_includes(tmp_path):
    parent=tmp_path/'parent.cfg';parent.write_text('avoidSgf3PatternDirs=games\navoidSgf3PatternUtility=.1\n')
    path=tmp_path/'source.cfg';path.write_text('@include parent.cfg\nnumSearchThreads=1\navoidSgf3PatternLambda=.9\n')
    c=load_engine_config(path,environ={})['analysis']
    assert json.loads(c['pattern_corpora'])==[dict(paths=[str(tmp_path/'games')],penalty=.1,decay=.9)]


def test_gtp_analysis_and_genmove_defaults_are_separate(tmp_path):
    path=tmp_path/'gtp.cfg';path.write_text('numSearchThreads=1\nvisitCapContemptPla=W\n')
    gtp=load_engine_config(path,environ={},gtp=True)['analysis']
    analysis=load_engine_config(path,environ={})['analysis']
    assert gtp['wide_root_noise']==0 and analysis['wide_root_noise']==.04
    assert gtp['analysis_wide_root_noise']==.04 and not gtp['analysis_ignore_pre_root_history']
    assert gtp['visit_cap_contempt_player']==-1
    assert gtp['gtp_default_time_control'] and 'gtp_default_time_control' not in analysis
    path.write_text(path.read_text()+'maxVisits=100\n')
    assert 'gtp_default_time_control' not in load_engine_config(path,environ={},gtp=True)['analysis']
