"""Full-game records, checked AlphaZero views, and a derived unique-row catalog."""
import json
from pathlib import Path
import sqlite3
import numpy as np
from .schema import CONTRACT_ID, RAW_DTYPES
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
    n, t, canvas = m["games"], m["rows"], m["canvas"]
    require(n > 0 and t > 0 and 5 <= canvas <= 25, "invalid shard dimensions")
    require(a["game_offsets"].shape == (n+1,) and a["observation_offsets"].shape == (n+1,), "offset shapes")
    lengths = np.diff(a["game_offsets"])
    require(a["game_offsets"][0] == a["observation_offsets"][0] == 0 and
            a["game_offsets"][-1] == t and a["observation_offsets"][-1] == t+n and
            (lengths > 0).all() and (np.diff(a["observation_offsets"]) == lengths+1).all(), "invalid offsets")
    expected = {"observations": (t+n, 6, canvas, canvas), "players": (t+n,),
                "policies": (t, canvas*canvas), "visits": (t, canvas*canvas)}
    for key in ("game_ids", "seeds", "sizes", "rules", "winners", "reasons"):
        expected[key] = (n,)
    for key in ("actions", "simulations", "temperatures", "rewards"):
        expected[key] = (t,)
    for key, shape in expected.items():
        require(a[key].shape == shape, f"{key}: shape mismatch")
    require(len(np.unique(a["game_ids"])) == n, "duplicate game IDs in shard")
    require(((a["sizes"] >= 5) & (a["sizes"] <= canvas) & (lengths <= a["sizes"]**2)).all(), "game sizes/lengths")
    require(np.isin(a["rules"], [0, 1, 2]).all() and np.isin(a["winners"], [-1, 0, 1]).all() and
            np.isin(a["reasons"], [0, 1, 2]).all(), "invalid rule/result")
    require((a["observations"] <= 1).all(), "nonbinary observations")
    for key in ("policies", "temperatures", "rewards"):
        require(np.isfinite(a[key]).all(), f"nonfinite {key}")
    require((a["temperatures"] >= 0).all() and (a["simulations"] > 0).all(), "invalid temperature/budget")
    totals = a["visits"].sum(1)
    require((a["visits"] >= 0).all() and (totals >= a["simulations"]).all(), "invalid completed visits")
    require(np.allclose(a["policies"], a["visits"]/totals[:, None], rtol=1e-5, atol=1e-7), "policy differs from completed visits")
    for i, size in enumerate(a["sizes"]):
        lo, hi = a["game_offsets"][i:i+2]
        ol, oh = a["observation_offsets"][i:i+2]
        obs, players = a["observations"][ol:oh], a["players"][ol:oh]
        require((players == np.where(np.arange(len(players)) % 2 == 0, 1, -1)).all(), "player alternation")
        mask = np.zeros((canvas, canvas), np.uint8); mask[:size, :size] = 1
        require((obs[:, 3] == mask).all() and (obs[:, 2] == (players[:, None, None] == 1)*mask).all(), "board/color planes")
        require((obs[:, 4] == (a["rules"][i] == 1)*mask).all() and
                (obs[:, 5] == (a["rules"][i] == 2)*mask).all(), "rule planes")
        require((obs * (1-mask)[None, None]).sum() == 0 and not obs[0, :2].any(), "padding/empty opening")
        board = np.zeros((canvas, canvas), np.int8)
        for j, action in enumerate(a["actions"][lo:hi]):
            require(0 <= action < canvas*canvas, "action outside canvas")
            y, x = divmod(int(action), canvas)
            require(mask[y, x] and board[y, x] == 0, "unsubmittable recorded action")
            require((obs[j, 0] == (board == players[j])).all() and
                    (obs[j, 1] == (board == -players[j])).all(), "observation/transition mismatch")
            legal = (mask & (board == 0)).flatten()
            require((a["visits"][lo+j][~legal.astype(bool)] == 0).all(), "search visited masked action")
            board[y, x] = players[j]
        require((obs[-1, 0] == (board == players[-1])).all() and (obs[-1, 1] == (board == -players[-1])).all(), "final observation")
        winner = a["winners"][i]
        require((a["rewards"][lo:hi-1] == 0).all() and a["rewards"][hi-1] == winner*players[-2], "terminal reward")
        require((winner == 0) == (a["reasons"][i] == 0), "draw/result reason mismatch")
        if winner == 0:
            require(hi-lo == size*size, "non-full draw")
        if a["reasons"][i] == 2:
            require(a["rules"][i] == 2 and winner == -1 and players[-2] == 1, "forbidden-loss metadata")


def training_view(a):
    indices, targets = [], []
    for i, (lo, hi) in enumerate(zip(a["game_offsets"][:-1], a["game_offsets"][1:])):
        ol = a["observation_offsets"][i]
        ix = np.arange(ol, ol+hi-lo)
        indices.append(ix)
        targets.append(a["winners"][i] * a["players"][ix])
    return {"obs": a["observations"][np.concatenate(indices)], "policy": a["policies"],
            "value": np.concatenate(targets).astype(np.float32)}


class Catalog:
    """SQLite is only a derived manifest; complete raw shards remain authoritative."""
    def __init__(self, run_dir, run_id, config_id):
        self.run_dir = Path(run_dir)
        self.run_id, self.config_id = run_id, config_id
        self.db = sqlite3.connect(self.run_dir / "catalog.sqlite")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS shards (id TEXT PRIMARY KEY, path TEXT UNIQUE, sha TEXT, meta TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS games (id TEXT PRIMARY KEY, shard TEXT, rows INTEGER)")

    def close(self):
        self.db.close()

    def scan(self, models):
        known = {row[0] for row in self.db.execute("SELECT path FROM shards")}
        for relative in known:
            if not (self.run_dir / relative).is_file():
                raise ValueError(f"Committed raw shard is missing: {relative}")
        for path in sorted((self.run_dir / "data").rglob("*.npz")):
            relative = str(path.relative_to(self.run_dir))
            if relative in known:
                continue
            with np.load(path, allow_pickle=False) as a:
                m = metadata(a)
                ids, offsets = a["game_ids"], a["game_offsets"]
            if (m["contract"] != CONTRACT_ID or m["run_id"] != self.run_id or m["config_id"] != self.config_id or
                    models.get(m["cycle_id"]) != m["model_id"] or len(ids) != m["games"] or
                    offsets.shape != (len(ids)+1,) or offsets[0] != 0 or offsets[-1] != m["rows"] or (np.diff(offsets) <= 0).any()):
                raise ValueError(f"Raw shard identity/count mismatch: {path}")
            with self.db:
                self.db.execute("INSERT INTO shards VALUES (?,?,?,?)", (m["shard_id"], relative, sha256(path), json.dumps(m)))
                for game_id, rows in zip(ids, np.diff(offsets)):
                    identity = f'{m["run_id"]}:{m["attempt_id"]}:{m["worker_id"]}:{int(game_id)}'
                    self.db.execute("INSERT INTO games VALUES (?,?,?)", (identity, m["shard_id"], int(rows)))

    def entries(self):
        result = [{"path": p, "sha256": h, "metadata": json.loads(m)} for p, h, m in
                  self.db.execute("SELECT path,sha,meta FROM shards")]
        return sorted(result, key=lambda x: (x["metadata"]["cycle_id"], x["metadata"]["created_ns"], x["metadata"]["shard_id"]))

    def counts(self, recent_games):
        rows, games = self.db.execute("SELECT coalesce(sum(rows),0),count(*) FROM games").fetchone()
        lengths = [x[0] for x in self.db.execute("SELECT rows FROM games ORDER BY rowid DESC LIMIT ?", (recent_games,))]
        return rows, games, float(np.mean(lengths)) if lengths else None
