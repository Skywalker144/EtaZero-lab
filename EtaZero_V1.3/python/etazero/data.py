"""Full-game records, checked AlphaZero views, and a derived unique-row catalog."""
import json
import hashlib
from pathlib import Path
import sqlite3
import numpy as np
from .schema import CONTRACT_ID, RAW_DTYPES, PLANES, GLOBALS, unpack_observations


def metadata(arrays):
    return json.loads(arrays["metadata"].tobytes().decode())


def game_row_ranges(a):
    """Resolve this file's span against full, game-ordered row multiplicities."""
    m = metadata(a)
    totals = np.add.reduceat(a['row_repeats'], a['game_offsets'][:-1]).astype(np.int64)
    totals += np.bincount(a['side_game_indices'], weights=a['side_row_repeats'], minlength=len(totals)).astype(np.int64)
    offsets = np.r_[0, np.cumsum(totals)]
    begin, count = m['row_begin'], m['rows']
    if (type(begin) is not int or type(count) is not int or begin < 0 or count < 0 or
            begin+count > offsets[-1] or begin > totals[0]):
        raise ValueError('Invalid shard training row span')
    starts = np.clip(begin-offsets[:-1], 0, totals)
    ends = np.clip(begin+count-offsets[:-1], 0, totals)
    # A represented nonempty game must contribute rows. Zero-row completed
    # games are retained as provenance, exactly once, without fabricated targets.
    if ((totals > 0) & (ends <= starts)).any() or int((ends-starts).sum()) != count:
        raise ValueError('Shard includes an unselected nonempty trajectory')
    return totals, starts, ends


def read_raw(path, *, deep=True):
    with np.load(path, allow_pickle=False) as file:
        arrays = {k: file[k] for k in file.files}
    validate_raw(arrays, str(path), deep=deep)
    return arrays


def validate_raw(a, source="record", *, deep=True):
    def require(ok, message):
        if not ok:
            raise ValueError(f"{source}: {message}")
    m = metadata(a)
    require(m.get('algorithm', 'alphazero') in ('alphazero', 'muzero'), 'unknown raw algorithm')
    muzero = m.get('algorithm') == 'muzero'
    extra = {'trajectory_policy', 'trajectory_q_values', 'trajectory_q_visits'} if muzero else set()
    require(set(a) == set(RAW_DTYPES) | extra, "raw fields do not match schema")
    for key, dtype in RAW_DTYPES.items():
        require(a[key].dtype == np.dtype(dtype), f"{key}: dtype mismatch")
    m = metadata(a)
    require(m["contract"] == CONTRACT_ID, "contract mismatch")
    n, t, canvas = m["games"], m["plies"], m["canvas"]
    require(n > 0 and t > 0 and 5 <= canvas <= 25, "invalid shard dimensions")
    require(a["game_offsets"].shape == (n+1,) and a["observation_offsets"].shape == (n+1,), "offset shapes")
    lengths = np.diff(a["game_offsets"])
    require(a["game_offsets"][0] == a["observation_offsets"][0] == 0 and
            a["game_offsets"][-1] == t and a["observation_offsets"][-1] == t+n and
            (lengths > 0).all() and (np.diff(a["observation_offsets"]) == lengths+1).all(), "invalid offsets")
    root_diagnostics = ('root_policy_invalid_mass_sum', 'root_policy_invalid_mass_count')
    if any(key in m for key in root_diagnostics):
        require(all(isinstance(m.get(key), list) and len(m[key]) == n for key in root_diagnostics),
                'root policy diagnostic shapes')
        require(all(type(count) is int and 0 <= count <= length-prefix
                    for count, length, prefix in zip(m[root_diagnostics[1]], lengths, a['opening_moves'])),
                'root policy diagnostic counts')
        require(all(type(total) in (int, float) and np.isfinite(total) and 0 <= total <= count
                    for total, count in zip(m[root_diagnostics[0]], m[root_diagnostics[1]])),
                'root policy diagnostic sums')
        require(all(count == 0 or count == length-prefix
                    for count, length, prefix in zip(m[root_diagnostics[1]], lengths, a['opening_moves'])),
                'root policy diagnostic must cover the searched suffix')
    samples = np.flatnonzero(a['row_repeats'] > 0)
    require(np.array_equal(a['sample_indices'], samples), "sample indices must match positive row repeats")
    s = len(samples)
    expected = {"observations": (t+n, len(PLANES), (canvas*canvas+7)//8), "players": (t+n,), "globals": (t+n,len(GLOBALS)),
                "sample_indices": (s,), "forbidden_input": (int(a["row_repeats"].sum()),), "policies": (s, canvas*canvas), "visits": (s, canvas*canvas),
                "opponent_policies": (s, canvas*canvas), "opponent_policy_weights": (s,),
                "q_values": (len(a['forbidden_input']),canvas*canvas), "q_visits": (s,canvas*canvas),
                "network_wdl": (t,3), "search_wdl": (t,3)}
    for key in ("game_ids", "seeds", "sizes", "rules", "winners", "reasons", "opening_moves", "balanced_moves", "policy_moves",
                "opening_attempts", "opening_status", "start_values", "initial_position_moves", "initial_position_kind", "hint_actions"):
        expected[key] = (n,)
    for key in ("actions", "simulations", "temperatures", "rewards", "train_mask", "row_repeats", "target_weights",
                "policy_surprises", "value_surprises", "cheap_search"):
        expected[key] = (t,)
    for key, shape in expected.items():
        require(a[key].shape == shape, f"{key}: shape mismatch")
    d = len(a['side_players'])
    side_shapes = {'side_observations': (d,len(PLANES),(canvas*canvas+7)//8),
                   'side_globals': (d,len(GLOBALS)), 'side_policies': (d,canvas*canvas),
                   'side_visits': (d,canvas*canvas), 'side_wdl': (d,3),
                   'side_q_values': (int(a['side_row_repeats'].sum()),canvas*canvas),
                   'side_q_visits': (d,canvas*canvas),
                   'side_forbidden_input': (int(a['side_row_repeats'].sum()),)}
    for key in ('side_game_indices','side_players','side_row_repeats','side_target_weights'):
        side_shapes[key] = (d,)
    for key, shape in side_shapes.items():
        require(a[key].shape == shape, f'{key}: shape mismatch')
    for key in ('reanalyzed','reanalysis_used_outcome','reanalysis_original_visits','reanalysis_policy_surprise','reanalysis_value_surprise'):
        require(a[key].shape == (t,), f'{key}: shape mismatch')
    if muzero:
        require(type(m.get('unroll_steps')) is int and 1 <= m['unroll_steps'] <= 32, 'invalid unroll length')
        for key in extra:
            require(a[key].shape == (t,canvas*canvas) and a[key].dtype == np.int16, f'{key}: layout mismatch')
    game_row_ranges(a)
    # Native writers publish complete immutable records. Replay reads check the
    # layout; trajectory reconstruction is an explicit offline/test operation.
    if not deep:
        return
    if muzero:
        from .muzero.data import validate_trajectory
        validate_trajectory(a)
    require(len(np.unique(a["game_ids"])) == n, "duplicate game IDs in shard")
    require(((a["sizes"] >= 5) & (a["sizes"] <= canvas) & (lengths <= a["sizes"]**2)).all(), "game sizes/lengths")
    require(np.isin(a["rules"], [0, 1, 2]).all() and np.isin(a["winners"], [-1, 0, 1]).all() and
            np.isin(a["reasons"], [0, 1, 2]).all(), "invalid rule/result")
    if canvas*canvas % 8:
        require((a["observations"][:, :, -1] & ((1 << (8-canvas*canvas % 8))-1) == 0).all(), "nonzero packed tail bits")
    for key in ("policies", "opponent_policies", "opponent_policy_weights", "temperatures", "rewards", "globals", "target_weights", "policy_surprises", "value_surprises", "network_wdl", "search_wdl"):
        require(np.isfinite(a[key]).all(), f"nonfinite {key}")
    require(np.isin(a["train_mask"], [0, 1]).all(), "invalid train mask")
    game_row_ranges(a)
    require((a["opening_moves"] >= 0).all() and (a["opening_moves"] <= lengths).all() and
            (a["balanced_moves"] >= 0).all() and (a["policy_moves"] >= 0).all() and
            (a["initial_position_moves"] >= 0).all() and
            (a["opening_moves"] == a["balanced_moves"]+a["policy_moves"]+a["initial_position_moves"]).all(), "opening lengths")
    kind=a['initial_position_kind'];hint=a['hint_actions']
    require(np.isin(kind,[0,1,2,3,4]).all(), 'initial position kind')
    require(((kind==0) <= (a['initial_position_moves']==0)).all() and
            ((kind>0) <= ((a['balanced_moves']==0) & (a['policy_moves']==0) & (a['opening_attempts']==0))).all(), 'initial prefix split')
    require(((kind==1)==(hint>=0)).all() and ((hint>=-1)&(hint<canvas*canvas)).all(), 'hint action/kind')
    require(((kind<2) | (a['initial_position_moves']>0)).all(), 'fork prefix length')
    require(np.isin(a["opening_status"], [0, 1, 2]).all() and (a["opening_attempts"] >= 0).all() and
            np.isfinite(a["start_values"]).all() and (np.abs(a["start_values"]) <= 1).all(), "opening result")
    require(((a["opening_status"] == 1) == (a["balanced_moves"] > 0)).all() and
            ((a["opening_status"] == 0) == (a["opening_attempts"] == 0)).all(), "opening status/attempts")
    require(len(m["opening_failures"]) == n and all(isinstance(x, str) for x in m["opening_failures"]), "opening failures")
    train = a["train_mask"].astype(bool)
    require((a["temperatures"] >= 0).all() and (a["simulations"][train] >= 0).all(), "invalid temperature/budget")
    require((a["simulations"][~train] == 0).all() and (a["temperatures"][~train] == 0).all(), "opening prefix has search supervision")
    totals = a["visits"].sum(1)
    require((a["visits"] >= 0).all() and (totals >= a["simulations"][samples]).all(), "invalid completed visits")
    require((totals > 0).all(), "search has no completed visits")
    require((a['policies'] >= 0).all() and (a['policies'].sum(1) > 0).all() and (a['policies'] <= 30000).all(), "invalid policy target")
    require((np.abs(a['q_values'].astype(np.int32))<=32000).all() and
            ((a['q_visits']>=0)&(a['q_visits']<=32000)).all(), 'invalid Q value/node visits')
    q_rows=np.repeat(np.arange(s),a['row_repeats'][samples])
    require(not a['q_values'][a['q_visits'][q_rows]==0].any(), 'Q value on unvisited child')
    require(((a['policies'] > 0) <= (a['visits'] > 0)).all(), "policy target on unvisited action")
    require((a['opponent_policies'] >= 0).all() and (a['opponent_policies'].sum(1) > 0).all() and (a['opponent_policies'] <= 30000).all(),
            "invalid opponent policy target")
    require(np.isin(a['opponent_policy_weights'], [0, 1]).all(), "invalid opponent policy weight")
    for key in ('network_wdl','search_wdl'):
        require((a[key] >= -1e-6).all() and (a[key] <= 1+1e-6).all() and
                np.allclose(a[key].sum(1),1,atol=1e-5), f"invalid {key}")
    require(np.isin(a['cheap_search'],[0,1]).all(), "invalid cheap search flag")
    require((a['target_weights'] >= 0).all() and (a['policy_surprises'] >= 0).all() and
            ((a['value_surprises'] >= 0)&(a['value_surprises'] <= 1)).all(), "invalid surprise weights")
    require((a['row_repeats'] >= 0).all() and (a['row_repeats'][~train] == 0).all() and
            (a['target_weights'][~train] == 0).all(), "invalid row repeats/prefix weights")
    # Stored float32 weights can round across an integer boundary.
    require((np.abs(a['row_repeats']-a['target_weights']) <= 1.00001).all(), "row repeats differ from randomized weight rounding")
    require(np.isin(a["forbidden_input"], [0,1]).all(), "invalid output-row forbidden flag")
    row_games = np.repeat(np.arange(n), np.add.reduceat(a["row_repeats"], a["game_offsets"][:-1]))
    require(not a["forbidden_input"][a["rules"][row_games] != 2].any(), "forbidden input outside Renju")
    for key in ("reanalyzed","reanalysis_used_outcome","reanalysis_original_visits","reanalysis_policy_surprise","reanalysis_value_surprise"):
        require(a[key].shape==(t,) and np.isfinite(a[key]).all(), f"invalid {key}")
    require(np.isin(a["reanalyzed"],[0,1]).all() and np.isin(a["reanalysis_used_outcome"],[0,1]).all() and
            (a["reanalyzed"]<=a["cheap_search"]).all() and (a["reanalyzed"]<=a["train_mask"]).all() and
            (a["reanalysis_used_outcome"]<=a["reanalyzed"]).all(), "invalid reanalysis flags")
    for key in ("reanalysis_original_visits","reanalysis_policy_surprise","reanalysis_value_surprise"):
        require((a[key]>=0).all() and not a[key][a["reanalyzed"]==0].any(), f"invalid reanalysis diagnostic: {key}")
    require((a["reanalysis_original_visits"][a["reanalyzed"]==1]>=1).all(), "missing original cheap visits")
    d = len(a['side_players'])
    side_shapes = {'side_observations': (d,len(PLANES),(canvas*canvas+7)//8),
                   'side_globals': (d,len(GLOBALS)), 'side_policies': (d,canvas*canvas),
                   'side_visits': (d,canvas*canvas), 'side_wdl': (d,3),
                   'side_q_values': (int(a['side_row_repeats'].sum()),canvas*canvas),
                   'side_q_visits': (d,canvas*canvas),
                   'side_forbidden_input': (int(a['side_row_repeats'].sum()),)}
    for key in ('side_game_indices','side_players','side_row_repeats','side_target_weights'):
        side_shapes[key] = (d,)
    for key, shape in side_shapes.items():
        require(a[key].shape == shape, f'{key}: shape mismatch')
    require(np.isin(a['side_players'], [-1,1]).all() and
            ((a['side_game_indices'] >= 0)&(a['side_game_indices'] < n)).all(), 'invalid side player/game')
    require((a['side_row_repeats'] >= 0).all() and np.isfinite(a['side_target_weights']).all() and
            (a['side_target_weights'] >= 0).all() and
            (np.abs(a['side_row_repeats']-a['side_target_weights']) <= 1.00001).all(), 'invalid side frequency')
    require(np.isfinite(a['side_wdl']).all() and (a['side_wdl'] >= 0).all() and
            np.allclose(a['side_wdl'].sum(1), 1, atol=1e-5), 'invalid side WDL')
    require((a['side_policies'] >= 0).all() and (a['side_policies'] <= 30000).all() and
            (a['side_policies'].sum(1) > 0).all() and (a['side_visits'] >= 0).all() and
            ((a['side_policies'] > 0) <= (a['side_visits'] > 0)).all(), 'invalid side policy/visits')
    require(np.isin(a['side_forbidden_input'], [0,1]).all(), 'invalid side forbidden flag')
    require((np.abs(a['side_q_values'].astype(np.int32))<=32000).all() and
            ((a['side_q_visits']>=0)&(a['side_q_visits']<=32000)).all(), 'invalid side Q value/node visits')
    sq_rows=np.repeat(np.arange(d),a['side_row_repeats'])
    require(not a['side_q_values'][a['side_q_visits'][sq_rows]==0].any(), 'side Q value on unvisited child')
    side_row_games = np.repeat(a['side_game_indices'], a['side_row_repeats'])
    require(not a['side_forbidden_input'][a['rules'][side_row_games] != 2].any(), 'side forbidden outside Renju')
    for i, g in enumerate(a['side_game_indices']):
        observation = unpack_observations(a['side_observations'][i:i+1],canvas)[0]
        size, rule, player = a['sizes'][g], a['rules'][g], a['side_players'][i]
        mask = np.zeros((canvas,canvas), np.uint8);mask[:size,:size] = 1
        require(np.array_equal(observation[0],mask) and not (observation*(1-mask)).any() and
                not (observation[1]*observation[2]).any(), 'invalid side board/padding')
        require(np.array_equal(a['side_globals'][i], [rule==1,rule==2,-player if rule==2 else 0,rule==2,0,0]),
                'side rule/color globals')
        require(not (observation[3:5]*(observation[1]+observation[2])).any() and
                not observation[4 if player==1 else 3].any() and
                (rule==2 or not observation[3:5].any()), 'invalid side forbidden perspective')
        legal = (mask & (observation[1]+observation[2]==0)).flatten().astype(bool)
        require(not a['side_visits'][i,~legal].any(), 'side visited masked action')
        require(not a['side_q_visits'][i,~legal].any(), 'side Q visited masked action')
    sample_lookup = {int(index): row for row, index in enumerate(samples)}
    for i, size in enumerate(a["sizes"]):
        lo, hi = a["game_offsets"][i:i+2]
        ol, oh = a["observation_offsets"][i:i+2]
        prefix = int(a["opening_moves"][i])
        require((a["train_mask"][lo:hi] == (np.arange(hi-lo) >= prefix)).all(), "opening prefix mask")
        # Validate one complete trajectory at a time; retain compact shard arrays.
        obs, players = unpack_observations(a["observations"][ol:oh], canvas), a["players"][ol:oh]
        require((players == np.where(np.arange(len(players)) % 2 == 0, 1, -1)).all(), "player alternation")
        if hint[i]>=0:
            hy,hx=divmod(int(hint[i]),canvas)
            require(hy<size and hx<size and not obs[prefix,1:3,hy,hx].any(), 'occupied or padded hint move')
        mask = np.zeros((canvas, canvas), np.uint8); mask[:size, :size] = 1
        require((obs[:, 0] == mask).all(), "board mask")
        globals = a["globals"][ol:oh]
        require(np.isin(globals[:,4], [0,1]).all() and np.isfinite(globals[:,5]).all() and
                (np.abs(globals[:,5])<=0.5*np.log2(100)+1e-6).all(), "invalid PDA globals")
        require((globals[:,4]==globals[0,4]).all() and
                np.allclose(globals[:,5]*players,globals[0,5]*players[0],rtol=0,atol=1e-7) and
                ((globals[:,4]==0)==(globals[:,5]==0)).all(), "PDA conditioning must persist and change sign with player")
        if kind[i]>=2:require(not globals[:,4:6].any(), 'fork positions must not inherit PDA')
        renju = a["rules"][i] == 2
        require((globals[:, 0] == (a["rules"][i] == 1)).all() and
                (globals[:, 1] == renju).all() and
                (globals[:, 2] == (-players if renju else 0)).all(), "rule/color globals")
        require(np.isin(globals[:, 3], [0, 1]).all() and
                (renju or not globals[:, 3].any()), "forbidden feature flag")
        require((obs * (1-mask)[None, None]).sum() == 0 and not obs[0, 1:3].any(), "padding/empty opening")
        require(not obs[globals[:, 3] == 0, 3:5].any(), "dropped forbidden planes")
        require(not obs[players == 1, 4].any() and not obs[players == -1, 3].any(), "forbidden perspective")
        require(not (obs[:, 3:5] * (obs[:, 1]+obs[:, 2])[:, None]).any(), "occupied forbidden feature")
        require((globals[:, 3] == renju).all(), "trajectory must retain full forbidden features")
        board = np.zeros((canvas, canvas), np.int8)
        for j, action in enumerate(a["actions"][lo:hi]):
            require(0 <= action < canvas*canvas, "action outside canvas")
            y, x = divmod(int(action), canvas)
            require(mask[y, x] and board[y, x] == 0, "unsubmittable recorded action")
            require((obs[j, 1] == (board == players[j])).all() and
                    (obs[j, 2] == (board == -players[j])).all(), "observation/transition mismatch")
            legal = (mask & (board == 0)).flatten()
            if muzero:
                require(not a['trajectory_policy'][lo+j][~legal.astype(bool)].any(), 'MuZero trajectory policy on occupied/padded point')
                require(not a['trajectory_q_visits'][lo+j][~legal.astype(bool)].any(), 'MuZero trajectory Q visited masked action')
            if lo+j in sample_lookup:
                require((a["visits"][sample_lookup[lo+j]][~legal.astype(bool)] == 0).all(), "search visited masked action")
                require(not a['q_visits'][sample_lookup[lo+j]][~legal.astype(bool)].any(), 'Q visited masked action')
            board[y, x] = players[j]
            if lo+j in sample_lookup:
                row = sample_lookup[lo+j]
                has_next = lo+j+1 < hi and not (a["reanalyzed"][lo+j] and not a["reanalysis_used_outcome"][lo+j])
                require(a['opponent_policy_weights'][row] == has_next, "opponent policy terminal weight")
                opponent = a['opponent_policies'][row]
                if has_next:
                    next_legal = (mask & (board == 0)).flatten().astype(bool)
                    require((opponent[~next_legal] == 0).all(), "opponent policy on occupied/padded successor action")
                    if lo+j+1 in sample_lookup:
                        require(np.array_equal(opponent, a['policies'][sample_lookup[lo+j+1]]),
                                "opponent policy differs from next turn search")
                else:
                    require((opponent == 1).all(),
                            "opponent policy terminal placeholder")
        require((obs[-1, 1] == (board == players[-1])).all() and (obs[-1, 2] == (board == -players[-1])).all(), "final observation")
        winner = a["winners"][i]
        require((a["rewards"][lo:hi-1] == 0).all() and a["rewards"][hi-1] == winner*players[-2], "terminal reward")
        require((winner == 0) == (a["reasons"][i] == 0), "draw/result reason mismatch")
        if winner == 0:
            require(hi-lo == size*size, "non-full draw")
        if a["reasons"][i] == 2:
            require(a["rules"][i] == 2 and winner == -1 and players[-2] == 1, "forbidden-loss metadata")


def trajectory_td_targets(search_wdl, players, terminal, area):
    """Finite horizon in fixed Black perspective, then restore each row's player."""
    fixed = search_wdl.astype(np.float64).copy()
    fixed[players == -1] = fixed[players == -1, ::-1]
    factors = 1.0 / (1.0 + area * np.array([0.176, 0.056, 0.016]))
    future = np.broadcast_to(terminal, (3, 3)).copy().astype(np.float64)
    targets = np.empty((len(fixed), 3, 3), np.float32)
    for i in range(len(fixed) - 1, -1, -1):
        future = factors[:, None] * fixed[i] + (1.0 - factors[:, None]) * future
        targets[i] = future if players[i] == 1 else future[:, ::-1]
    return targets


TRAIN_TARGETS = ('policy', 'opponent_policy', 'opponent_policy_weight', 'value',
                 'td_value', 'full_game_weight', 'q_values', 'q_visits')


def training_targets(config):
    if (config.get('agent', {}).get('algorithm') == 'muzero' and
            not config.get('muzero_training', {}).get('auxiliary_losses', True)):
        return ('policy', 'value')
    return tuple(k for k in TRAIN_TARGETS if config['network'].get('predict_q_values', False)
                 or k not in ('q_values', 'q_visits'))


def training_view(a, targets=TRAIN_TARGETS):
    if metadata(a).get('algorithm') == 'muzero':
        from .muzero.data import training_view as sequence_view
        return sequence_view(a, targets)
    return alphazero_training_view(a, targets)


def alphazero_training_view(a, targets=TRAIN_TARGETS):
    # Each sampled position is stored once; multiplicity is resolved only here.
    samples = a['sample_indices']
    rows = np.repeat(np.arange(len(samples)), a['row_repeats'][samples])
    ix = samples[rows]
    games = np.searchsorted(a['game_offsets'][1:], ix, side='right')
    obs_ix = a['observation_offsets'][games] + ix-a['game_offsets'][games]
    result = a['winners'][games] * a['players'][obs_ix]
    td = np.empty((len(a['actions']), 3, 3), np.float32) if 'td_value' in targets else None
    for g, size in enumerate(a['sizes']) if td is not None else ():
        lo, hi = a['game_offsets'][g:g+2]
        ol = a['observation_offsets'][g]
        winner = a['winners'][g]
        terminal = np.array([winner == 1, winner == 0, winner == -1], np.float64)
        td[lo:hi] = trajectory_td_targets(a['search_wdl'][lo:hi], a['players'][ol:ol+hi-lo], terminal, int(size)**2)
    obs = a["observations"][obs_ix]
    globals = a["globals"][obs_ix]
    dropped = a["forbidden_input"] == 0
    obs[dropped, 3:5] = 0
    globals[dropped, 3] = 0
    # KataGo suppresses outcome-derived auxiliary targets on these reanalysis
    # rows, independently of the main WDL/TD targets. Resolve trajectory flags
    # through ix so every repeated output row receives the same gate.
    full_game_weight = ((a['reanalyzed'][ix] == 0) |
                        (a['reanalysis_used_outcome'][ix] != 0)).astype(np.float32)
    main = {"obs": obs, "globals": globals,
            "policy": a['policies'][rows].astype(np.float32),
            "value": np.stack((result==1,result==0,result==-1),axis=1).astype(np.float32)}
    if 'opponent_policy' in targets:
        main.update(opponent_policy=a['opponent_policies'][rows].astype(np.float32),
                    opponent_policy_weight=a['opponent_policy_weights'][rows])
    if 'td_value' in targets:
        main.update(td_value=td[ix], full_game_weight=full_game_weight)
    if 'q_values' in targets:
        main.update(q_values=a['q_values'].astype(np.float32)/32000.0,
                    q_visits=a['q_visits'][rows].astype(np.float32))

    side_rows = np.repeat(np.arange(len(a['side_players'])), a['side_row_repeats'])
    side_obs, side_globals = a['side_observations'][side_rows], a['side_globals'][side_rows]
    dropped = a['side_forbidden_input'] == 0
    side_obs[dropped,3:5] = 0;side_globals[dropped,3] = 0
    side_value = a['side_wdl'][side_rows]
    side = {'obs': side_obs, 'globals': side_globals, 'policy': a['side_policies'][side_rows].astype(np.float32),
            'value': side_value}
    if 'opponent_policy' in targets:
        side.update(opponent_policy=np.ones((len(side_rows),main['policy'].shape[1]),np.float32),
                    opponent_policy_weight=np.zeros(len(side_rows),np.float32))
    if 'td_value' in targets:
        side.update(td_value=np.repeat(side_value[:,None],3,axis=1),
                    full_game_weight=np.zeros(len(side_rows),np.float32))
    if 'q_values' in targets:
        side.update(q_values=a['side_q_values'].astype(np.float32)/32000.0,
                    q_visits=a['side_q_visits'][side_rows].astype(np.float32))
    totals, starts, ends = game_row_ranges(a)
    # Source writes each finished game's main rows followed by its side rows.
    # Select after constructing full-horizon targets, so a shard boundary never
    # changes TD, opponent gates, Q rounding or forbidden dropout.
    side_games = a['side_game_indices'][side_rows]
    main_offsets = np.r_[0,np.cumsum(np.bincount(games,minlength=len(totals)))]
    order = np.concatenate([np.r_[np.arange(main_offsets[g],main_offsets[g+1]), len(ix)+np.flatnonzero(side_games == g)]
                            for g in range(len(totals))])
    m = metadata(a)
    order = order[m['row_begin']:m['row_begin']+m['rows']]
    return {key: np.concatenate((main[key],side[key]))[order] for key in main}


def game_weight_statistics(a):
    """Distinct trainable main-position sums, independent of shard row spans."""
    m = metadata(a)
    if m.get('algorithm') != 'muzero': return []
    result=[]
    for g,game in enumerate(a['game_ids']):
        lo,hi=a['game_offsets'][g:g+2]
        valid=a['train_mask'][lo:hi].astype(bool)
        result.append({'id':f'{m["run_id"]}:{m["attempt_id"]}:{m["worker_id"]}:{int(game)}',
                       'sum':float(a['target_weights'][lo:hi][valid].sum(dtype=np.float64)),
                       'count':int(valid.sum())})
    return result


class Catalog:
    """SQLite is only a derived manifest; complete raw shards remain authoritative."""
    def __init__(self, run_dir, run_id, config_id):
        self.run_dir = Path(run_dir)
        self.run_id, self.config_id = run_id, config_id
        (self.run_dir/".internal").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / ".internal/catalog.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute('CREATE TABLE IF NOT EXISTS shards (id TEXT PRIMARY KEY, path TEXT UNIQUE, meta TEXT, weights TEXT, '
                        'iteration INTEGER, created INTEGER, rows INTEGER, stats TEXT, mtime INTEGER, size INTEGER)')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(shards)')}
        if columns != {'id','path','meta','weights','iteration','created','rows','stats','mtime','size'}:
            self.db.close()
            raise ValueError('Derived catalog schema differs; rebuild catalog.sqlite from the raw shards')
        self.db.execute('CREATE INDEX IF NOT EXISTS shard_timeline ON shards(iteration,created,id)')
        self.db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, rows INTEGER, total INTEGER, fingerprint TEXT)")
        if {row[1] for row in self.db.execute('PRAGMA table_info(games)')} != {'id','rows','total','fingerprint'}:
            self.db.close()
            raise ValueError('Derived catalog schema differs; rebuild catalog.sqlite from the raw shards')
        self.db.execute('CREATE TABLE IF NOT EXISTS game_segments (game TEXT, start INTEGER, end INTEGER, shard TEXT, PRIMARY KEY(game,start))')
        self.db.execute('CREATE TABLE IF NOT EXISTS totals (id INTEGER PRIMARY KEY CHECK(id=1), rows INTEGER, games INTEGER)')
        self.db.execute('INSERT OR IGNORE INTO totals VALUES (1,0,0)')
        self.db.execute('CREATE INDEX IF NOT EXISTS shard_mtime ON shards(mtime DESC)')
        self.db.commit()
        self.random_rows = self.db.execute("SELECT COALESCE(SUM(rows),0) FROM shards WHERE substr(json_extract(meta,'$.model_id'),1,7)='random:'").fetchone()[0]

    def close(self):
        self.db.close()

    def scan(self, models, directories=None):
        # A complete reconciliation is needed at startup/recovery only. During a
        # live run, the controller supplies the current iteration's producer directory.
        if directories is None:
            for (relative,) in self.db.execute('SELECT path FROM shards'):
                if not (self.run_dir/relative).is_file():
                    raise ValueError(f'Committed raw shard is missing: {relative}')
            directories = [self.run_dir/'selfplay']
        paths = sorted(path for directory in directories for path in Path(directory).rglob('*.npz'))
        for path in paths:
            relative = str(path.relative_to(self.run_dir))
            stat=path.stat()
            indexed=self.db.execute('SELECT mtime,size FROM shards WHERE path=?',(relative,)).fetchone()
            if indexed:
                mtime,size=indexed
                if (mtime,size)!=(stat.st_mtime_ns,stat.st_size):
                    if stat.st_size != size:raise ValueError(f'Committed raw shard changed size: {path}')
                    with self.db:self.db.execute('UPDATE shards SET mtime=?,size=? WHERE path=?',(stat.st_mtime_ns,stat.st_size,relative))
                continue
            a = read_raw(path, deep=False)
            m = metadata(a)
            ids, offsets, prefix = a["game_ids"], a["game_offsets"], a["opening_moves"]
            winners, status = a['winners'], a['opening_status']
            totals, starts, ends = game_row_ranges(a)
            game_rows = ends-starts
            owners = starts == 0
            fingerprints=[]
            for g in range(len(ids)):
                lo,hi=offsets[g:g+2];ol,oh=a['observation_offsets'][g:g+2]
                digest=hashlib.sha256()
                for key in ('seeds','sizes','rules','winners','reasons','opening_moves','opening_status','opening_attempts',
                            'balanced_moves','policy_moves','initial_position_moves','initial_position_kind','hint_actions'):
                    digest.update(a[key][g:g+1].tobytes())
                for key in ('actions','row_repeats','search_wdl','network_wdl'):
                    digest.update(a[key][lo:hi].tobytes())
                for key in ('observations','globals'):
                    digest.update(a[key][ol:oh].tobytes())
                if 'root_policy_invalid_mass_count' in m:
                    digest.update(np.asarray(m['root_policy_invalid_mass_sum'][g:g+1], dtype='<f8').tobytes())
                    digest.update(np.asarray(m['root_policy_invalid_mass_count'][g:g+1], dtype='<u8').tobytes())
                if m.get('algorithm') == 'muzero':
                    digest.update(str(m['unroll_steps']).encode())
                    for key in ('trajectory_policy','trajectory_q_values','trajectory_q_visits',
                                'target_weights','train_mask','reanalyzed','reanalysis_used_outcome'):
                        digest.update(a[key][lo:hi].tobytes())
                fingerprints.append(digest.hexdigest())
            weights = game_weight_statistics(a)
            stats = {'games':int(owners.sum()),'rows':m['rows'],'plies':int(np.diff(offsets)[owners].sum()),
                     'black_wins':int((winners[owners]==1).sum()),'white_wins':int((winners[owners]==-1).sum()),
                     'draws':int((winners[owners]==0).sum()),'opening_moves':int(prefix[owners].sum()),
                     'balanced_games':int((status[owners]==1).sum()),'opening_failures':int((status[owners]==2).sum()),
                     'opening_attempts':int(a['opening_attempts'][owners].sum()),'policy_moves':int(a['policy_moves'][owners].sum())}
            if 'root_policy_invalid_mass_count' in m:
                stats.update(root_policy_invalid_mass_sum=float(np.asarray(m['root_policy_invalid_mass_sum'])[owners].sum()),
                             root_policy_invalid_mass_count=int(np.asarray(m['root_policy_invalid_mass_count'])[owners].sum()))
            if (m["contract"] != CONTRACT_ID or m["run_id"] != self.run_id or
                    models.get(m["iteration_id"]) != m["model_id"] or len(ids) != m["games"] or
                    offsets.shape != (len(ids)+1,) or offsets[0] != 0 or offsets[-1] != m["plies"] or prefix.shape != (len(ids),) or
                    (np.diff(offsets) <= 0).any() or (prefix < 0).any() or (prefix > np.diff(offsets)).any() or
                    int(game_rows.sum()) != m["rows"]):
                raise ValueError(f"Raw shard identity/count mismatch: {path}")
            with self.db:
                self.db.execute('INSERT INTO shards VALUES (?,?,?,?,?,?,?,?,?,?)',
                                (m['shard_id'],relative,json.dumps(m),json.dumps(weights),m['iteration_id'],m['created_ns'],m['rows'],json.dumps(stats),stat.st_mtime_ns,stat.st_size))
                added_games=0
                for game_id, rows, total, start, end, fingerprint in zip(ids,game_rows,totals,starts,ends,fingerprints):
                    identity = f'{m["run_id"]}:{m["attempt_id"]}:{m["worker_id"]}:{int(game_id)}'
                    existing=self.db.execute('SELECT total,fingerprint FROM games WHERE id=?',(identity,)).fetchone()
                    if existing and existing!=(int(total),fingerprint):
                        raise ValueError(f'Conflicting trajectory fragments: {identity}')
                    # The primary key rejects exact duplicates; the interval
                    # check also rejects overlapping fragments with new starts.
                    self.db.execute('INSERT INTO game_segments VALUES (?,?,?,?)',(identity,int(start),int(end),m['shard_id']))
                    overlap=self.db.execute('SELECT 1 FROM game_segments WHERE game=? AND shard<>? AND start<? AND end>?',
                                            (identity,m['shard_id'],int(end),int(start))).fetchone()
                    if overlap:raise ValueError(f'Overlapping training row fragments: {identity}')
                    if existing:self.db.execute('UPDATE games SET rows=rows+? WHERE id=?',(int(rows),identity))
                    else:
                        self.db.execute('INSERT INTO games VALUES (?,?,?,?)',(identity,int(rows),int(total),fingerprint));added_games+=1
                self.db.execute('UPDATE totals SET rows=rows+?,games=games+? WHERE id=1',(m['rows'],added_games))
            if m['model_id'].startswith('random:'):
                self.random_rows += m['rows']

    def entries(self, window_rows=None):
        result, rows = [], 0
        for p,m,w,n,mtime in self.db.execute('SELECT path,meta,weights,rows,mtime FROM shards ORDER BY mtime DESC,rowid DESC'):
            result.append({'path':p,'metadata':json.loads(m),'weight_stats':json.loads(w),'mtime_ns':mtime});rows += n
            if window_rows is not None and rows >= window_rows:
                break
        return result[::-1]

    def counts(self):
        return self.db.execute('SELECT rows,games FROM totals WHERE id=1').fetchone()

    def replay_counts(self, minimum):
        rows, random_rows = self.counts()[0], self.random_rows
        postrandom = rows-random_rows
        return dict(raw_rows=rows,random_rows=random_rows,postrandom_rows=postrandom,
                    usable_rows=min(random_rows,minimum)+postrandom)

    def previous_rows_per_game(self, iteration):
        """Pool actual rows and unique games from the preceding two rounds."""
        rows, games = self.db.execute('SELECT COALESCE(SUM(rows),0), '
                                     'COALESCE(SUM(json_extract(stats, "$.games")),0) '
                                     'FROM shards WHERE iteration>=? AND iteration<?',
                                     (max(0, iteration-2), iteration)).fetchone()
        return rows/games if games else None

    def iteration_counts(self, iteration):
        rows, games = self.db.execute('SELECT COALESCE(SUM(rows),0), COALESCE(SUM(json_extract(stats, "$.games")),0) '
                                      'FROM shards WHERE iteration=?', (iteration,)).fetchone()
        return rows, games

    def statistics(self, iteration):
        result = {"games": 0, "rows": 0, "plies": 0, "black_wins": 0, "white_wins": 0, "draws": 0,
                  "opening_moves": 0, "balanced_games": 0, "opening_attempts": 0, "policy_moves": 0,
                  "opening_failures": 0}
        for (payload,) in self.db.execute('SELECT stats FROM shards WHERE iteration=?', (iteration,)):
            for key, value in json.loads(payload).items():
                result[key] = result.get(key, 0) + value
        for name, key in (("avg_rows_per_game","rows"),("avg_game_length","plies")):
            result[name] = result[key]/result['games'] if result['games'] else 0
        if result.get('root_policy_invalid_mass_count', 0):
            result['root_policy_invalid_mass'] = result['root_policy_invalid_mass_sum']/result['root_policy_invalid_mass_count']
        return result
