"""Execute fixed Mctx numerical functions with NumPy arrays, without a JAX runtime.

Only imports/type annotations/assertion helpers are adapted. The source's Q,
action-selection and visit-sequence function bodies execute verbatim.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
import numpy as np
from scipy.special import softmax


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--binary',type=Path,default=Path(__file__).resolve().parents[2]/'build/gumbel_test')
    args=parser.parse_args()
    reference=json.loads((Path(__file__).resolve().parents[2]/'reference_sources.json').read_text())['Mctx']
    def equal_shape(arrays):
        assert len({np.shape(a) for a in arrays})==1
    def shape(arrays,expected):
        for a in arrays if isinstance(arrays,list) else [arrays]:assert np.shape(a)==tuple(expected)
    scope=dict(math=math,jnp=np,jax=SimpleNamespace(nn=SimpleNamespace(softmax=lambda a:softmax(a,axis=-1))),
               chex=SimpleNamespace(assert_shape=shape,assert_equal_shape=equal_shape))
    def load(path,names):
        data=(args.source/path).read_bytes()
        assert hashlib.sha256(data).hexdigest()==reference['sha256'][path],f'Unregistered reference: {path}'
        functions=[n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name in names]
        future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        module=ast.fix_missing_locations(ast.Module(body=[future,*functions],type_ignores=[]))
        exec(compile(module,path,'exec'),scope)
    load('mctx/_src/qtransforms.py',{'qtransform_completed_by_mix_value','_rescale_qvalues','_complete_qvalues','_compute_mixed_value'})
    scope['qtransforms']=SimpleNamespace(qtransform_completed_by_mix_value=scope['qtransform_completed_by_mix_value'])
    load('mctx/_src/seq_halving.py',{'score_considered','get_sequence_of_considered_visits','get_table_of_considered_visits'})
    scope['seq_halving']=SimpleNamespace(**{name:scope[name] for name in ('score_considered','get_sequence_of_considered_visits','get_table_of_considered_visits')})
    load('mctx/_src/action_selection.py',{'gumbel_muzero_root_action_selection','masked_argmax'})
    outputs=json.loads(subprocess.check_output([str(args.binary),'--reference'],text=True))
    for actual in outputs:
        counts=np.zeros((1,3),dtype=np.int32); logits=np.log(np.array([[.5,.3,.2]]))
        tree=SimpleNamespace(children_visits=counts,children_prior_logits=logits,raw_values=np.array([.1]),
                             root_invalid_actions=np.zeros(3,dtype=np.int32),extra_data=SimpleNamespace(root_gumbel=np.zeros(3)),
                             qvalues=lambda index:np.array([.8,-.2,-.6]))
        for _ in range(actual['budget']):
            action=scope['gumbel_muzero_root_action_selection'](None,tree,np.int32(0),num_simulations=actual['budget'],max_num_considered_actions=16)
            counts[0,action]+=1
        q=scope['qtransform_completed_by_mix_value'](tree,np.int32(0))
        scores=scope['score_considered'](counts.max(),np.zeros(3),logits[0],q,counts[0])
        assert actual['action']==int(np.argmax(scores))
        np.testing.assert_array_equal(actual['visits'],counts[0])
        np.testing.assert_allclose(actual['policy'],softmax(logits[0]+q),rtol=1e-12,atol=1e-14)
    print(f"{len(outputs)} budgets: C++ counts, SH winner and dense target agree with fixed Mctx")


if __name__=='__main__':main()
