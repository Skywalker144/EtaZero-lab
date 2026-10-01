# Reference code and attribution

KataGo is the primary reference for game-thread scheduling, shared batch inference,
bounded queues, pooled search synchronization, atomic statistics, tiered child storage,
parallel tree reclamation, direct-mapped NN caches,
multiple inference servers, packed binary observations, reusable writer buffers, background data writing, and selfplay → shuffle → train → export file
boundaries. The checked source is commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`.
The power-law window function and per-group keep-target subsampling in
`python/etazero/shuffle.py` are adapted from KataGo's `python/shuffle.py`;
the two-phase shuffle, count-and-slice partitioning, multi-wave
processing and queue services follow its architecture. The CUDA upload prefetch stream,
authenticated training-view cache, restorable snapshot eviction and persistent worker
protocol are EtaZero adaptations, not copied from its Python reader.
KataGo's copyright and MIT license are retained in [licenses/KataGo.txt](licenses/KataGo.txt).

The recursive Renju analyzer in `cpp/src/game/rules.cpp` is adapted from the user's
MuZero_V2 working tree, preserving exact-five priority, overline, distinct-four and
recursive live-three semantics. MuZero_V2 supplies the plotting theme/layout and thin-script experiment scheduler reference, as adapted in `python/etazero/plotting.py` and `experiment.py`. SkyZero V7.19/V8.1 informs umbrella discovery, GPU queuing and resumable experiment usage. Shared initialization here means weights only, with independent random bootstrap per arm. MuZero_V2 supplies the deterministic random evaluator and cold-start quota reference and informed the opening/data integration boundary. It is also a reference for the TorchScript /
LibTorch boundary. EtaZero's input feature contract follows the scoped SkyZero_V7.19 adaptation described below.

[reference_sources.json](reference_sources.json) records the actual inspected source
paths and SHA-256 values. These directories are development references; builds and
runs use only files and dependencies inside EtaZero and the selected Python environment.

Balanced opening in `cpp/src/selfplay/opening.cpp` is checked directly against
KataGomo's `cpp/game/randomopening.cpp` (no-VC selfplay path), with policy initialization
from `cpp/program/playutils.cpp`. It retains the skeleton distribution, positional
weights, both root perspectives, candidate balance weights, terminal rejection and
fallback retry behavior. EtaZero uses its own seeded RNG, five-plane/four-global WDL-value
interface and cancellation, and removes the source's negative sampling epsilon to
avoid selecting zero-weight actions at probability endpoints. Opening prefixes remain
in full trajectories but are excluded from training rows. SkyZero V8.1's
`gomokuopening.cpp` and MuZero V2 were inspected as integration cross-checks;
KataGomo is the mechanism's authoritative source. Its MIT license is retained in
[licenses/KataGomo.txt](licenses/KataGomo.txt).

The reference scope is the stated runtime/data architecture, opening and Renju semantics.
Go rules, score utility and auxiliary learning heads, distributed training services and
its multiple native inference backends are outside the first-round implementation.


Independent match scheduling, per-opening/per-game resume and joint Elo fitting in
`python/etazero/arena.py`, `elo.py`, and `process.py`, with the streaming native
match command in `cpp/src/commands/main.cpp`, are adapted from MuZero_V2's arena,
Elo and match implementations. The paired-opening bootstrap, shared first-player
advantage, weak Gaussian prior and rating anchor are retained. Model discovery is
adapted to EtaZero's committed iteration records and manifests; native execution
uses AlphaZero PUCT and EtaZero's batched evaluator instead of MuZero recurrent
inference. Evaluation visits count the root initialization (99 fresh edge
simulations for 100v), following the KataGo root-visit budget convention; this
explicitly differs from MuZero_V2's edge-simulation count. FPU, LCB and policy pruning are explicit independent profile fields. Training commit timing follows
MuZero_V2's complete-iteration boundary; discarded products are archived rather
than deleted. KataGo's match command and example configuration inform independent
match profiles, model identities and game/evaluator concurrency.

SkyZero_V7.19 supplies the input-feature reference (`cpp/envs/gomoku.h`, `cpp/selfplay_manager.h`, `python/nets.py`): five spatial planes, the first four global features, global linear projection, and per-training-row forbidden-feature dropout on an independent RNG stream. Draw utility and PDA are outside this adaptation. Renju feature calculation uses EtaZero's existing rule analyzer, not a second rule implementation.

Training D4 in `python/etazero/symmetry.py` follows the exact eight-transform numbering and batch-level sampling of SkyZero_V8.1 `katago/python/katago/train/data_processing_pytorch.py`. Optimizer grouping, BN learning-rate/weight-decay policies, warmup, Lookahead and SWA in `optimization.py` follow its native `katago/python/train.py`, model regularization groups and norm metrics. EtaZero retains its own architecture and mean loss reporting, uses batch-sum backward, checkpointable augmentation RNG and slow weights, and refreshes LR/WD each update. Optimizer groups and BN head assignments are adapted to the current residual network. KataGomo `python/train.py` and `python/train.sh` were inspected for optimizer settings; its older SGD decay coefficients and SWA-scale-1 launcher are research references rather than active defaults. These adaptations retain the KataGo MIT license.

WDL, FPU visited-policy interpolation/reduction, shaped Dirichlet concentration,
forced root playouts, inverse-PUCT target pruning and LCB selection in
`cpp/src/search/search.cpp` follow KataGo's `searchexplorehelpers.cpp`,
`searchhelpers.cpp` and `searchresults.cpp`. Selfplay cap randomization, its
clear/preserve-tree and remove-root-noise branches, disabling LCB for selfplay
moves while retaining it in targets, and surprise weight redistribution follow
`cpp/program/play.cpp`; row multiplicity follows `cpp/dataio/trainingwrite.cpp`.
Extreme-value Reduce Visits and its priority relative to PCR, fixed-perspective
history window, quadratic cap/target-weight interpolation and full-search tree
behavior in `cpp/src/selfplay/search_limits.cpp` follow the same `play.cpp`.
Baseline Reduce Visits parameters follow SkyZero_V8.1's selfplay configuration.
EtaZero caps include the initial root evaluation and require at least one child
visit; its WDL-only utility does not need the source's dynamic-score-center presearch.
SkyZero_V8.1's same native paths and MuZero_V2 search/training targets were
inspected as cross-checks. Weighted subtree value updates, weighted LCB ESS,
PUCT's 0.01 offset and optional log/stdev scaling, virtual sample weights, root
D4 probability averaging and NN/root/move temperatures follow the native SkyZero
and KataGo search update/NN helpers and evaluator. The t(3) CDF uses a mathematically
equivalent closed form to generate the source's 2000-point interpolation grid.
Transformed complete inputs identify orientation-specific cached outputs; single
evaluations use the identity symmetry rather than source-option random orientation.
EtaZero retains its own RNG and architecture, terminal WDL supervision, and full-game
raw records with persisted stochastic row repeats. Go score utility, uncertainty
weighting, noise pruning, subtree value bias, graph search, reanalysis and auxiliary
heads are outside this adaptation. Draw is a real Gomoku draw rather than KataGo's
no-result channel.
