"""Full-game records, checked AlphaZero views, and a derived unique-row catalog."""
import json
from pathlib import Path
import sqlite3
import numpy as np
from .schema import CONTRACT_ID, RAW_DTYPES, PLANES, GLOBALS, unpack_observations
from .storage import sha256


def metadata(arrays):
    return json.loads(arrays["metadata"].tobytes().decode())


def read_raw(path):
    with np.load(path, allow_pickle=False) as file:
        arrays = {k: file[k] for k in file.files}
    validate_raw(arrays, str(path))
    return arrays


def validate_raw(a, source="record"):
    def require(ok, message):
        if not ok:
            raise ValueError(f"{source}: {message}")
    require(set(a) == set(RAW_DTYPES), "raw fields do not match schema")
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
    samples = np.flatnonzero(a['row_repeats'] > 0)
    require(np.array_equal(a['sample_indices'], samples), "sample indices must match positive row repeats")
    s = len(samples)
    expected = {"observations": (t+n, len(PLANES), (canvas*canvas+7)//8), "players": (t+n,), "globals": (t+n,len(GLOBALS)),
                "sample_indices": (s,), "policies": (s, canvas*canvas), "visits": (s, canvas*canvas),
                "opponent_policies": (s, canvas*canvas), "opponent_policy_weights": (s,),
                "network_wdl": (t,3), "search_wdl": (t,3)}
    for key in ("game_ids", "seeds", "sizes", "rules", "winners", "reasons", "opening_moves", "balanced_moves", "policy_moves",
                "opening_attempts", "opening_status", "start_values"):
        expected[key] = (n,)
    for key in ("actions", "simulations", "temperatures", "rewards", "train_mask", "row_repeats", "target_weights",
                "policy_surprises", "value_surprises", "cheap_search"):
        expected[key] = (t,)
    for key, shape in expected.items():
        require(a[key].shape == shape, f"{key}: shape mismatch")
    require(len(np.unique(a["game_ids"])) == n, "duplicate game IDs in shard")
    require(((a["sizes"] >= 5) & (a["sizes"] <= canvas) & (lengths <= a["sizes"]**2)).all(), "game sizes/lengths")
    require(np.isin(a["rules"], [0, 1, 2]).all() and np.isin(a["winners"], [-1, 0, 1]).all() and
            np.isin(a["reasons"], [0, 1, 2]).all(), "invalid rule/result")
    if canvas*canvas % 8:
        require((a["observations"][:, :, -1] & ((1 << (8-canvas*canvas % 8))-1) == 0).all(), "nonzero packed tail bits")
    for key in ("policies", "opponent_policies", "opponent_policy_weights", "temperatures", "rewards", "globals", "target_weights", "policy_surprises", "value_surprises", "network_wdl", "search_wdl"):
        require(np.isfinite(a[key]).all(), f"nonfinite {key}")
    require(np.isin(a["train_mask"], [0, 1]).all() and int(a["row_repeats"].sum()) == m["rows"], "effective row count/mask")
    require((a["opening_moves"] >= 0).all() and (a["opening_moves"] <= lengths).all() and
            (a["balanced_moves"] >= 0).all() and (a["policy_moves"] >= 0).all() and
            (a["opening_moves"] == a["balanced_moves"]+a["policy_moves"]).all(), "opening lengths")
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
    require((a['policies'] >= 0).all() and np.allclose(a['policies'].sum(1), 1, atol=1e-6), "invalid policy target")
    require(((a['policies'] > 0) <= (a['visits'] > 0)).all(), "policy target on unvisited action")
    require((a['opponent_policies'] >= 0).all() and np.allclose(a['opponent_policies'].sum(1), 1, atol=1e-6),
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
    sample_lookup = {int(index): row for row, index in enumerate(samples)}
    for i, size in enumerate(a["sizes"]):
        lo, hi = a["game_offsets"][i:i+2]
        ol, oh = a["observation_offsets"][i:i+2]
        prefix = int(a["opening_moves"][i])
        require((a["train_mask"][lo:hi] == (np.arange(hi-lo) >= prefix)).all(), "opening prefix mask")
        # Validate one complete trajectory at a time; retain compact shard arrays.
        obs, players = unpack_observations(a["observations"][ol:oh], canvas), a["players"][ol:oh]
        require((players == np.where(np.arange(len(players)) % 2 == 0, 1, -1)).all(), "player alternation")
        mask = np.zeros((canvas, canvas), np.uint8); mask[:size, :size] = 1
        require((obs[:, 0] == mask).all(), "board mask")
        globals = a["globals"][ol:oh]
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
        require((globals[np.r_[np.arange(prefix), len(globals)-1], 3] == renju).all(),
                "dropout outside training rows")
        board = np.zeros((canvas, canvas), np.int8)
        for j, action in enumerate(a["actions"][lo:hi]):
            require(0 <= action < canvas*canvas, "action outside canvas")
            y, x = divmod(int(action), canvas)
            require(mask[y, x] and board[y, x] == 0, "unsubmittable recorded action")
            require((obs[j, 1] == (board == players[j])).all() and
                    (obs[j, 2] == (board == -players[j])).all(), "observation/transition mismatch")
            legal = (mask & (board == 0)).flatten()
            if lo+j in sample_lookup:
                require((a["visits"][sample_lookup[lo+j]][~legal.astype(bool)] == 0).all(), "search visited masked action")
            board[y, x] = players[j]
            if lo+j in sample_lookup:
                row = sample_lookup[lo+j]
                has_next = lo+j+1 < hi
                require(a['opponent_policy_weights'][row] == has_next, "opponent policy terminal weight")
                opponent = a['opponent_policies'][row]
                if has_next:
                    next_legal = (mask & (board == 0)).flatten().astype(bool)
                    require((opponent[~next_legal] == 0).all(), "opponent policy on occupied/padded successor action")
                    if lo+j+1 in sample_lookup:
                        require(np.array_equal(opponent, a['policies'][sample_lookup[lo+j+1]]),
                                "opponent policy differs from next turn search")
                else:
                    require(np.allclose(opponent, 1/(canvas*canvas), rtol=0, atol=1e-7),
                            "opponent policy terminal placeholder")
        require((obs[-1, 1] == (board == players[-1])).all() and (obs[-1, 2] == (board == -players[-1])).all(), "final observation")
        winner = a["winners"][i]
        require((a["rewards"][lo:hi-1] == 0).all() and a["rewards"][hi-1] == winner*players[-2], "terminal reward")
        require((winner == 0) == (a["reasons"][i] == 0), "draw/result reason mismatch")
        if winner == 0:
            require(hi-lo == size*size, "non-full draw")
        if a["reasons"][i] == 2:
            require(a["rules"][i] == 2 and winner == -1 and players[-2] == 1, "forbidden-loss metadata")


def training_view(a):
    # Each sampled position is stored once; multiplicity is resolved only here.
    samples = a['sample_indices']
    rows = np.repeat(np.arange(len(samples)), a['row_repeats'][samples])
    ix = samples[rows]
    games = np.searchsorted(a['game_offsets'][1:], ix, side='right')
    obs_ix = a['observation_offsets'][games] + ix-a['game_offsets'][games]
    result = a['winners'][games] * a['players'][obs_ix]
    return {"obs": a['observations'][obs_ix], "globals": a['globals'][obs_ix],
            "policy": a['policies'][rows],
            "opponent_policy": a['opponent_policies'][rows],
            "opponent_policy_weight": a['opponent_policy_weights'][rows],
            "value": np.stack((result==1,result==0,result==-1),axis=1).astype(np.float32)}


class Catalog:
    """SQLite is only a derived manifest; complete raw shards remain authoritative."""
    def __init__(self, run_dir, run_id, config_id):
        self.run_dir = Path(run_dir)
        self.run_id, self.config_id = run_id, config_id
        (self.run_dir/".internal").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.run_dir / ".internal/catalog.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute('CREATE TABLE IF NOT EXISTS shards (id TEXT PRIMARY KEY, path TEXT UNIQUE, sha TEXT, meta TEXT, '
                        'iteration INTEGER, created INTEGER, rows INTEGER, stats TEXT)')
        columns = {row[1] for row in self.db.execute('PRAGMA table_info(shards)')}
        if columns != {'id','path','sha','meta','iteration','created','rows','stats'}:
            self.db.close()
            raise ValueError('Derived catalog schema differs; rebuild catalog.sqlite from the raw shards')
        self.db.execute('CREATE INDEX IF NOT EXISTS shard_timeline ON shards(iteration,created,id)')
        self.db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, shard TEXT, rows INTEGER)")
        self.db.execute('CREATE TABLE IF NOT EXISTS totals (id INTEGER PRIMARY KEY CHECK(id=1), rows INTEGER, games INTEGER)')
        self.db.execute('INSERT OR IGNORE INTO totals VALUES (1,0,0)')
        self.db.commit()

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
            if self.db.execute('SELECT 1 FROM shards WHERE path=?',(relative,)).fetchone():
                continue
            with np.load(path, allow_pickle=False) as a:
                m = metadata(a)
                ids, offsets, prefix = a["game_ids"], a["game_offsets"], a["opening_moves"]
                winners, status = a['winners'], a['opening_status']
                repeats = a['row_repeats']
                game_rows = np.add.reduceat(repeats, offsets[:-1])
                stats = {'games':len(ids),'rows':m['rows'],'plies':m['plies'],
                         'black_wins':int((winners==1).sum()),'white_wins':int((winners==-1).sum()),
                         'draws':int((winners==0).sum()),'opening_moves':int(prefix.sum()),
                         'balanced_games':int((status==1).sum()),'opening_failures':int((status==2).sum()),
                         'opening_attempts':int(a['opening_attempts'].sum()),'policy_moves':int(a['policy_moves'].sum())}
            if (m["contract"] != CONTRACT_ID or m["run_id"] != self.run_id or m["config_id"] != self.config_id or
                    models.get(m["iteration_id"]) != m["model_id"] or len(ids) != m["games"] or
                    offsets.shape != (len(ids)+1,) or offsets[0] != 0 or offsets[-1] != m["plies"] or prefix.shape != (len(ids),) or
                    (np.diff(offsets) <= 0).any() or (prefix < 0).any() or (prefix > np.diff(offsets)).any() or
                    int(game_rows.sum()) != m["rows"]):
                raise ValueError(f"Raw shard identity/count mismatch: {path}")
            with self.db:
                self.db.execute('INSERT INTO shards VALUES (?,?,?,?,?,?,?,?)',
                                (m['shard_id'],relative,sha256(path),json.dumps(m),m['iteration_id'],m['created_ns'],m['rows'],json.dumps(stats)))
                for game_id, rows in zip(ids, game_rows):
                    identity = f'{m["run_id"]}:{m["attempt_id"]}:{m["worker_id"]}:{int(game_id)}'
                    self.db.execute("INSERT INTO games VALUES (?,?,?)", (identity, m["shard_id"], int(rows)))
                self.db.execute('UPDATE totals SET rows=rows+?,games=games+? WHERE id=1',(m['rows'],m['games']))

    def entries(self, window_rows=None):
        result, rows = [], 0
        for p,h,m,n in self.db.execute('SELECT path,sha,meta,rows FROM shards ORDER BY iteration DESC,created DESC,id DESC'):
            result.append({'path':p,'sha256':h,'metadata':json.loads(m)});rows += n
            if window_rows is not None and rows >= window_rows:
                break
        return result[::-1]

    def counts(self, recent_games):
        rows, games = self.db.execute('SELECT rows,games FROM totals WHERE id=1').fetchone()
        lengths = [x[0] for x in self.db.execute("SELECT rows FROM games ORDER BY rowid DESC LIMIT ?", (recent_games,))]
        return rows, games, float(np.mean(lengths)) if lengths else None

    def iteration_counts(self, iteration):
        rows, games = self.db.execute('SELECT COALESCE(SUM(rows),0), COALESCE(SUM(json_extract(meta, "$.games")),0) '
                                      'FROM shards WHERE iteration=?', (iteration,)).fetchone()
        return rows, games

    def statistics(self, iteration):
        result = {"games": 0, "rows": 0, "plies": 0, "black_wins": 0, "white_wins": 0, "draws": 0,
                  "opening_moves": 0, "balanced_games": 0, "opening_attempts": 0, "policy_moves": 0,
                  "opening_failures": 0}
        for (payload,) in self.db.execute('SELECT stats FROM shards WHERE iteration=?', (iteration,)):
            for key, value in json.loads(payload).items():
                result[key] += value
        for name, key in (("avg_rows_per_game","rows"),("avg_game_length","plies")):
            result[name] = result[key]/result['games'] if result['games'] else 0
        return result
