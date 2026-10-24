"""Explicit engine profiles, shared settings, precedence and standalone loading."""
from pathlib import Path
import shutil
import subprocess

import pytest
from config_samples import CONFIGS
from etazero.config import ROOT, fingerprint, load_config
from etazero.engine_config import load_engine_config, load_match_opening_config


@pytest.fixture
def profiles(tmp_path):
    shutil.copytree(CONFIGS/'baseline', tmp_path/'base')
    (tmp_path/'base/run.cfg').unlink()
    child=tmp_path/'independent';child.mkdir()
    (child/'engine.cfg').write_text('@include ../base/engine.cfg\n[engine]\nvisits=37\n')
    for mode in ('analysis','match'):
        (child/f'{mode}.cfg').write_text(f'@include ../base/{mode}.cfg\n@include engine.cfg\n')
    return child


def test_shared_parameters_and_mode_overrides_without_training_files(profiles):
    for match,mode in ((False,'analysis'),(True,'match')):
        assert load_engine_config(profiles,match,environ={})[mode]['visits']==37
        assert load_engine_config(profiles/f'{mode}.cfg',match,environ={})==load_engine_config(profiles,match,environ={})
    (profiles/'run.cfg').write_text('This is deliberately not a valid training configuration.')
    (profiles/'analysis.cfg').write_text('@include ../base/analysis.cfg\n@include engine.cfg\n[analysis]\nvisits=53\n')
    assert load_engine_config(profiles,environ={})['analysis']['visits']==53
    assert load_engine_config(profiles,True,environ={})['match']['visits']==37
    assert not load_engine_config(profiles,environ={})['analysis']['reuse_tree']
    assert load_engine_config(profiles,True,environ={})['match']['reuse_tree']


def test_local_and_environment_precedence_and_scope(profiles):
    # An included profile's machine-local values never leak into a child.
    (profiles.parent/'base/analysis.cfg.local').write_text('[analysis]\nvisits=999\n')
    (profiles/'engine.cfg.local').write_text('[engine]\nvisits=41\n')
    assert load_engine_config(profiles,environ={})['analysis']['visits']==41
    assert load_engine_config(profiles,True,environ={})['match']['visits']==41
    (profiles/'analysis.cfg.local').write_text('[analysis]\nvisits=43\n')
    assert load_engine_config(profiles,environ={})['analysis']['visits']==43
    env={'ENGINE_VISITS':'47','ANALYSIS_VISITS':'51','MATCH_VISITS':'59','SELFPLAY_VISITS':'999'}
    assert load_engine_config(profiles,environ=env)['analysis']['visits']==51
    assert load_engine_config(profiles,True,environ=env)['match']['visits']==59
    assert load_engine_config(profiles,environ={'ENGINE_VISITS':'47'})['analysis']['visits']==47


@pytest.mark.parametrize('mode', ['analysis', 'match'])
def test_local_includes_keep_file_override_order_across_engine_and_mode(profiles, mode):
    (profiles/'common.cfg').write_text(f'[engine]\nvisits=17\n[{mode}]\nvisits=23\n')
    local = profiles/f'{mode}.cfg.local'
    local.write_text('@include common.cfg\n[engine]\nvisits=31\n')
    assert load_engine_config(profiles, mode == 'match', environ={})[mode]['visits'] == 31
    local.write_text(f'@include common.cfg\n[{mode}]\nvisits=37\n[engine]\nvisits=41\n')
    assert load_engine_config(profiles, mode == 'match', environ={})[mode]['visits'] == 41


def test_training_identity_does_not_read_engine_files(tmp_path,monkeypatch):
    shutil.copytree(CONFIGS,tmp_path/'configs')
    monkeypatch.setattr('etazero.config.ROOT',tmp_path)
    selected=tmp_path/'configs/smoke_test'
    before=load_config(selected)
    (selected/'engine.cfg.local').write_text('[engine]\nvisits=77\ndevice=cpu\n')
    assert fingerprint(load_config(selected))==fingerprint(before)
    assert load_engine_config(selected,environ={})['analysis']['device']=='cpu'


@pytest.mark.parametrize('contents,error',[
    ('@include absent.cfg\n','No such file'),
    ('@include analysis.cfg\n','include cycle'),
    ('@include\n','Invalid include'),
    ('@include engine.cfg\n[engine]\ngames=4\n','Unknown key'),
    ('@include engine.cfg\n[DEFAULT]\nvisits=3\n','DEFAULT fields'),
    ('@include engine.cfg\n[analysis]\nvisits=nan\n','analysis.visits'),
])
def test_invalid_includes_and_fields_are_diagnostic(profiles,contents,error):
    (profiles/'analysis.cfg').write_text(contents)
    with pytest.raises((ValueError,FileNotFoundError),match=error):
        load_engine_config(profiles,environ={})


def test_opening_read_does_not_require_valid_search_overrides(profiles):
    env={'MATCH_VISITS':'invalid','MATCH_GAMES':'3','ANALYSIS_OPENING_BALANCE_EXPONENT':'99'}
    assert load_match_opening_config(profiles,environ=env)==load_engine_config(profiles,True,environ={})['opening']


@pytest.mark.parametrize('match', [False, True])
def test_go_graph_repetition_bound_default_and_overrides(profiles, match):
    mode = 'match' if match else 'analysis'
    key = mode.upper() + '_GRAPH_SEARCH_REP_BOUND'
    assert load_engine_config(profiles, match, environ={})[mode]['graph_search_rep_bound'] == 11
    for bound in (3, 17, 50):
        assert load_engine_config(profiles, match, environ={key: str(bound)})[mode]['graph_search_rep_bound'] == bound
    for bound in ('2', '51', '3.5'):
        with pytest.raises(ValueError, match='graph_search_rep_bound'):
            load_engine_config(profiles, match, environ={key: bound})


def test_cli_loads_standalone_profile_and_requires_model_source(profiles):
    result=subprocess.run(['bash',str(ROOT/'scripts/run.sh'),'analysis','--config',str(profiles/'analysis.cfg')],
                          text=True,capture_output=True)
    assert result.returncode!=0 and 'requires --model or --run-dir' in result.stderr
    help_result=subprocess.run(['bash',str(ROOT/'scripts/run.sh'),'analysis','--help'],text=True,capture_output=True)
    assert help_result.returncode==0 and '--stream' in help_result.stdout and '--config ' in help_result.stdout


def test_explicit_missing_source_does_not_use_defaults(tmp_path):
    with pytest.raises(FileNotFoundError,match='source does not exist'):
        load_engine_config(tmp_path/'typo.cfg',True,environ={})


def test_sibling_experiment_files_do_not_change_match_routing(profiles):
    before = load_engine_config(profiles, True, environ={})
    for name in ('exp.cfg', 'exp.cfg.local', 'elo.cfg'):
        (profiles/name).write_text('Not a valid configuration.\n')
    assert load_engine_config(profiles, True, environ={}) == before
    assert load_engine_config(profiles/'match.cfg', True, environ={}) == before


@pytest.mark.parametrize('section', ['experiment', 'elo'])
def test_match_rejects_orchestration_sections(profiles, section):
    with (profiles/'match.cfg').open('a') as stream:
        stream.write(f'\n[{section}]\n')
    with pytest.raises(ValueError, match='Wrong section/file ownership'):
        load_engine_config(profiles, True, environ={})


def test_final_overrides_precede_environment_and_are_validated(profiles):
    result = load_engine_config(profiles, True, environ={'MATCH_GAMES': 'invalid'},
                                overrides={'match': {'games': 12, 'game_threads': 3, 'reuse_tree': False}})
    assert result['match']['games'] == 12 and result['match']['game_threads'] == 3
    assert result['match']['reuse_tree'] is False
    with pytest.raises(ValueError, match='Match games'):
        load_engine_config(profiles, True, environ={}, overrides={'match': {'games': 0}})
    with pytest.raises(ValueError, match='Unknown engine override'):
        load_engine_config(profiles, True, environ={}, overrides={'match': {'unknown': 1}})


def test_match_resignation_fields_and_bounds(profiles):
    c=load_engine_config(profiles,True,environ={},overrides={'match':dict(allow_resignation=True,resign_threshold=-.9,resign_consec_turns=4)})
    assert c['match']['allow_resignation'] and c['match']['resign_threshold']==-.9 and c['match']['resign_consec_turns']==4
    for key,value in [('resign_threshold',.1),('resign_threshold',-1.1),('resign_consec_turns',0),('resign_consec_turns',101)]:
        with pytest.raises(ValueError,match='resign'):
            load_engine_config(profiles,True,environ={},overrides={'match':{key:value}})
