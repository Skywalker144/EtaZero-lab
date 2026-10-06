"""Unroll/module diagnostics retain commit, overflow and missing-data semantics."""
import json
import numpy as np

from etazero.plotting import METRICS, run_history, training_figure
from etazero.storage import save_json


def test_diagnostic_means_use_only_measured_committed_updates(tmp_path):
    save_json(tmp_path/'.internal/state.json',dict(iteration=4,checkpoint=dict(id='c',path='checkpoints/c.pt')))
    save_json(tmp_path/'checkpoints/c.json',dict(parent=None,committed_updates=['old','a','b','skip','latest']))
    def update(identity,iteration,**kwargs):
        return dict(dict.fromkeys(METRICS,1.),event='update',update_id=identity,iteration=iteration,**kwargs)
    a=update('a',1,step_losses=[2.,4.],grad_norms=dict(representation=3.,dynamics=4.,prediction=12.))
    b=update('b',1,step_losses=[4.,8.],grad_norms=dict(representation=6.,dynamics=8.,prediction=24.))
    skipped=update('skip',1,amp_skipped=True,step_losses=[6.,12.],grad_norms=dict.fromkeys(a['grad_norms'],float('inf')))
    latest=update('latest',2,amp_skipped=True,step_losses=[8.,16.],grad_norms=dict.fromkeys(a['grad_norms'],float('inf')))
    events=[update('old',1),a,a,b,skipped,latest,update('abandoned',1,step_losses=[999.,999.]),
            dict(event='iteration_complete',iteration=1,unique_rows=10),
            dict(event='iteration_complete',iteration=2,unique_rows=20)]
    path=tmp_path/'logs/events.jsonl';path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    first,last=run_history(tmp_path)
    assert first['steps']==4 and first['step_loss_updates']==3
    assert first['step_losses']==[4.,8.]
    assert first['module_gradient_steps']==2
    assert first['grad_norms']==dict(representation=4.5,dynamics=6.,prediction=18.)
    assert 'grad_norms' not in last
    figure=training_figure([first,last])
    assert len(figure.axes)==8 and 'MuZero' in figure._suptitle.get_text()
    mean,latest=figure.axes[6].lines
    np.testing.assert_array_equal(mean.get_xdata(),[0,1])
    np.testing.assert_array_equal(mean.get_ydata(),[6.,12.])
    np.testing.assert_array_equal(latest.get_ydata(),[8.,16.])
    for line,value in zip(figure.axes[7].lines,[4.5,6.,18.]):
        np.testing.assert_equal(line.get_ydata(),[value,np.nan])
    figure.clear()


def test_muzero_bootstrap_and_old_logs_show_missing_diagnostics():
    old=dict(dict.fromkeys(METRICS,1.),iteration=1,steps=1,phases={})
    for rows in ([],[old]):
        figure=training_figure(rows,algorithm='muzero')
        assert len(figure.axes)==8 and 'MuZero' in figure._suptitle.get_text()
        for axis in figure.axes[6:]:
            assert not axis.lines
            assert axis.texts[0].get_text()=='No measurements yet'
        figure.clear()
    figure=training_figure([old],algorithm='alphazero')
    assert len(figure.axes)==6
    figure.clear()
