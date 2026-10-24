"""Unroll diagnostics retain commit, overflow and missing-data semantics."""
import json
import numpy as np

from etazero.evaluation.plotting import METRICS, round_history, training_figure
from etazero.storage import save_json


def test_diagnostic_means_use_only_measured_committed_updates(tmp_path):
    save_json(tmp_path/'.internal/state.json',dict(iteration=4,checkpoint=dict(id='c',path='checkpoints/c.pt')))
    def update(identity,iteration,**kwargs):
        return dict(dict.fromkeys(METRICS,1.),event='update',step={'old':1,'a':2,'b':3,'skip':4,'latest':1,'abandoned':5}[identity],iteration=iteration,**kwargs)
    a=update('a',1,step_losses=[2.,4.])
    b=update('b',1,step_losses=[4.,8.])
    skipped=update('skip',1,amp_skipped=True,step_losses=[6.,12.])
    latest=update('latest',2,amp_skipped=True,step_losses=[8.,16.])
    events=[update('old',1),a,a,b,skipped,latest,update('abandoned',1,step_losses=[999.,999.]),
            dict(event='iteration_complete',iteration=1,unique_rows=10,total_samples=64),
            dict(event='iteration_complete',iteration=2,unique_rows=20,total_samples=80)]
    from test_experiment import journal
    journal(tmp_path, events)
    first,last=[round_history(tmp_path,i,n,0) for i,n in ((1,4),(2,1))]
    assert first['steps']==4 and first['step_loss_updates']==3
    assert first['step_losses']==[4.,8.]
    assert 'grad_norms' not in last
    figure=training_figure([first,last])
    assert len(figure.axes)==7 and 'MuZero' in figure._suptitle.get_text()
    mean,latest=figure.axes[6].lines
    np.testing.assert_array_equal(mean.get_xdata(),[0,1])
    np.testing.assert_array_equal(mean.get_ydata(),[6.,12.])
    np.testing.assert_array_equal(latest.get_ydata(),[8.,16.])
    figure.clear()


def test_muzero_bootstrap_and_old_logs_show_missing_diagnostics():
    old=dict(dict.fromkeys(METRICS,1.),iteration=1,steps=1,total_samples=32,phases={})
    for rows in ([],[old]):
        figure=training_figure(rows,algorithm='muzero')
        assert len(figure.axes)==7 and 'MuZero' in figure._suptitle.get_text()
        for axis in figure.axes[6:]:
            assert not axis.lines
            assert axis.texts[0].get_text()=='No measurements yet'
        figure.clear()
    figure=training_figure([old],algorithm='alphazero')
    assert len(figure.axes)==6
    figure.clear()
