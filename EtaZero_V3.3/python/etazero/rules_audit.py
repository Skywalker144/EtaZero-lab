"""Offline replay audit for history-dependent games (no second Python rules engine)."""
from .games import spec_for_rule
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .schema import RULES, unpack_observations


def validate_games(a, require):
    from .data import metadata
    m=metadata(a); canvas=m['canvas']; planes=a['observations'].shape[1]
    binary=Path(os.environ.get('ETAZERO_RULES_REPLAY', Path(__file__).resolve().parents[2]/'build/rules_replay'))
    if not binary.is_file():
        raise ValueError('Deep Go/Xiangqi auditing needs build/rules_replay (or ETAZERO_RULES_REPLAY)')
    samples={int(t):i for i,t in enumerate(a['sample_indices'])}
    for g,size in enumerate(a['sizes']):
        lo,hi=a['game_offsets'][g:g+2];ol,oh=a['observation_offsets'][g:g+2]
        globals=a['globals'][ol:oh]; rule=int(a['rules'][g])
        # KataGo feature 16 stores signed half doubling advantage; Xiangqi uses 4.
        pda_index=16 if rule==4 else 4
        signed=float(globals[0,pda_index])*2
        player=int(a['players'][ol])
        request=dict(size=int(size),canvas=canvas,rule=RULES[rule],actions=a['actions'][lo:hi].tolist(),
                     options=m['rule_options'][g],pda=abs(signed),pda_player=player*(1 if signed>0 else -1) if signed else 0)
        process=subprocess.run([str(binary)],input=json.dumps(request)+'\n',text=True,capture_output=True,check=False)
        require(process.returncode==0,'rule replay failed: '+process.stderr.strip())
        states=[json.loads(row) for row in process.stdout.splitlines()]
        require(len(states)==hi-lo+1,'rule replay length')
        obs=unpack_observations(a['observations'][ol:oh],canvas)
        for j,state in enumerate(states):
            expected=np.asarray(state['observation'],np.float32)
            require(np.array_equal(obs[j].reshape(-1),expected[:planes*canvas*canvas]),'rule observation mismatch')
            require(np.allclose(globals[j],expected[planes*canvas*canvas:],rtol=0,atol=1e-6),'rule globals mismatch')
            require(state['player']==a['players'][ol+j],'rule player mismatch')
            if j==hi-lo:break
            require(not state['finished'],'move after terminal')
            legal=np.zeros(a['policies'].shape[-1],bool);legal[state['legal']]=True
            require(legal[a['actions'][lo+j]],'illegal recorded move')
            if lo+j in samples:
                k=samples[lo+j]
                for key in ('policies','visits','q_visits'):
                    require(not a[key][k,~legal].any(),'search target on illegal move')
            if 'trajectory_policy' in a:
                require(not a['trajectory_policy'][lo+j,~legal].any(),'unroll policy on illegal move')
        final=states[-1]
        require(final['finished'] and final['winner']==a['winners'][g] and final['reason']==a['reasons'][g],'terminal result mismatch')
        require(not a['rewards'][lo:hi-1].any() and a['rewards'][hi-1]==final['winner']*a['players'][oh-2],'terminal reward mismatch')
        require((a['train_mask'][lo:hi] == (np.arange(hi-lo)>=a['opening_moves'][g])).all(),'opening prefix mask')
    # Side records have no transition history. Audit their static board and
    # targets here; their search legality is tested against the native rules.
    for i,g in enumerate(a['side_game_indices']):
        pda_index=15 if RULES[a['rules'][g]]=='go' else 3
        require(not a['side_globals'][i,pda_index:pda_index+2].any(),'side rows must clear PDA conditioning')
        obs=unpack_observations(a['side_observations'][i:i+1],canvas)[0]
        size=int(a['sizes'][g]);width=spec_for_rule(RULES[a['rules'][g]]).width(size)
        mask=np.zeros((canvas,canvas),np.uint8);mask[:size,:width]=1
        if RULES[a['rules'][g]]=='go':mask=unpack_observations(a['observations'][a['observation_offsets'][g]:a['observation_offsets'][g]+1],canvas)[0,0].astype(np.uint8)
        require(np.array_equal(obs[0],mask) and not (obs*(1-mask)).any(),'side mask/padding')
