# Reference code and attribution

`python/etazero/muzero/data.py` and `training.py` adapt MuZero_V2's unroll, absorbing-state, replay-weight and hidden/loss gradient boundaries to EtaZero's complete heads, row multiplicity, masked side continuations and batch-sum optimizer convention. The isolated latent search reuses EtaZero's configured PUCT and subtree aggregation formulas, rather than claiming exact MuZero_V2 search parity.

`python/etazero/muzero/network.py` uses the pinned MuZero_V2 masked min/max
normalization and gradient-scaling definitions, checked against the unchanged
source bodies by `tests/reference/check_muzero_network.py`. Its three independently
sized NBT trunks, five-plane/six-global input and complete heads reuse EtaZero's
KataGo-derived modules; they are not the V2 masked ResNet architecture. Latent
normalization explicitly computes in FP32 under autocast. MuZero parameter roles
are defined separately and reuse EtaZero's fson schedule. The independent native
backend implements initial/recurrent inference and device-local latent ownership;
AlphaZero's cache, batching and graph-search code is not transplanted into it.

The native hint loader links the selected environment's OpenSSL Crypto library
to verify SHA-256 over the exact bytes it parses. OpenSSL is a build dependency;
no cryptographic implementation is copied into EtaZero.

KataGo is the primary reference for game-thread scheduling, shared batch inference,
bounded queues, pooled search synchronization, atomic statistics, tiered child storage,
graph transpositions, catch-up edge visits, private promoted roots, parallel node reclamation, direct-mapped NN caches,
multiple inference servers, packed binary observations, reusable writer buffers, background data writing, and selfplay → shuffle → train → export file
boundaries. The checked source is commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`.
The power-law window function and per-group keep-target subsampling in
`python/etazero/shuffle.py` are adapted from KataGo's `python/shuffle.py`;
`python/etazero/reader.py` adapts the gap-delaying file order from
`python/katago/utils/training_data_generator.py`;
the two-phase shuffle, count-and-slice partitioning, multi-wave
processing and queue services follow its architecture. The CUDA upload prefetch stream,
authenticated training-view cache, restorable snapshot eviction and persistent worker
protocol are EtaZero adaptations, not copied from its Python reader.
KataGo's copyright and MIT license are retained in [licenses/KataGo.txt](licenses/KataGo.txt).

The recursive Renju analyzer in `cpp/src/game/rules.cpp` is adapted from the user's
MuZero_V2 working tree, preserving exact-five priority, overline, distinct-four and
recursive live-three semantics. Its forbidden outcomes and categories are independently
checked against the fixed KataGomo `CForbiddenPointFinder` by
`tests/reference/check_katagomo_rules.py`; this optional validation compiles the external
reference, while normal builds remain self-contained. MuZero_V2 supplies the plotting theme/layout and thin-script experiment scheduler reference, as adapted in `python/etazero/plotting.py` and `experiment.py`. SkyZero V7.19/V8.1 informs umbrella discovery, GPU queuing and resumable experiment usage. Shared initialization here means weights only, with independent random bootstrap per arm. MuZero_V2 supplies the deterministic random evaluator and cold-start quota reference and informed the opening/data integration boundary. It is also a reference for the TorchScript /
LibTorch boundary. EtaZero's input feature contract follows the scoped SkyZero_V7.19 adaptation described below.

The repository-level `web/` interface, HTTP session handling and persistent process
wrapper are adapted from the user's MuZero `web/` working tree and V2 `engine.py`.
EtaZero adds published-model manifest/checksum validation, canvas-aware controls and
search W/D/L display; its native `serve` command uses EtaZero Game and PUCT directly.
MuZero's `eval_main.cpp` supplies the session protocol and analysis JSON reference.

[reference_sources.json](reference_sources.json) records the actual inspected source
paths and SHA-256 values. These directories are development references; builds and
runs use only files and dependencies inside EtaZero and the selected Python environment.

Balanced opening in `cpp/src/selfplay/opening.cpp` is checked directly against
KataGomo's `cpp/game/randomopening.cpp` (no-VC selfplay path), with policy initialization
from `cpp/program/playutils.cpp`. It retains the skeleton distribution, positional
weights, both root perspectives, candidate balance weights, terminal rejection and
fallback retry behavior. EtaZero uses its own seeded RNG, five-plane/six-global WDL-value
interface and cancellation, and removes the source's negative sampling epsilon to
avoid selecting zero-weight actions at probability endpoints. Opening prefixes remain
in full trajectories but are excluded from training rows. SkyZero V8.1's
`gomokuopening.cpp` and MuZero V2 were inspected as integration cross-checks;
KataGomo is the mechanism's authoritative source. Its MIT license is retained in
[licenses/KataGomo.txt](licenses/KataGomo.txt).

The reference scope is the stated runtime/data architecture, opening and Renju semantics.
Go rules, score utility and Go-specific auxiliary heads, distributed training services and
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

SkyZero_V7.19 supplies the input-feature reference (`cpp/envs/gomoku.h`, `cpp/selfplay_manager.h`, `python/nets.py`): five spatial planes, the first four global features, global linear projection, and per-training-row forbidden-feature dropout on an independent RNG stream. Draw utility is outside this adaptation; the final two globals use KataGo PDA flag/signed half-d semantics. Renju feature calculation uses EtaZero's existing rule analyzer, not a second rule implementation.

Training D4 in `python/etazero/symmetry.py` follows the exact eight-transform numbering and batch-level sampling of SkyZero_V8.1 `katago/python/katago/train/data_processing_pytorch.py`. Optimizer grouping, BN learning-rate/weight-decay policies, warmup, Lookahead and SWA in `optimization.py` follow its native `katago/python/train.py`, model regularization groups and norm metrics. EtaZero uses its NBT/fson architecture and mean loss reporting, uses batch-sum backward, checkpointable augmentation RNG and slow weights, and refreshes LR/WD each update. Optimizer groups follow NBT/fson roles, including zero-centered gamma offsets and output-group final normalization. KataGomo `python/train.py` and `python/train.sh` were inspected for optimizer settings; its older SGD decay coefficients and SWA-scale-1 launcher are research references rather than active defaults. These adaptations retain the KataGo MIT license.

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
Canonical complete inputs, NN temperature and effective optimism identify cached outputs;
single evaluations randomize orientation on cache misses, while root ensembles bypass cache.
EtaZero retains its own RNG and architecture, terminal WDL supervision, and full-game
raw records with persisted stochastic row repeats. Go score utility, subtree value bias and Go-specific auxiliary
heads are outside this adaptation. Draw is a real Gomoku draw rather than KataGo's
no-result channel.

The six policy outputs in `network.py` follow KataGo `model_pytorch.py` head
ordering and `metrics_pytorch.py` targets: next actual turn opponent policy,
0.15 opponent coefficient, epsilon-plus-fourth-root softening over on-board
cells, and `train.py` default soft weight 8. `record.cpp` follows
`trainingwrite.cpp` successor availability independently of successor row sampling.
Export exposes primary and short-term optimistic policy, main WDL and real error
standard deviation, retaining the six-filter convolution shape.
EtaZero omits Go pass, uses source-style int16 policy quantization,
and logs weighted mean contributions for all policy heads. Other auxiliary
KataGo heads and metadata-only soft-policy filtering are outside this scope.

The NBT2 trunk, global policy head, size-conditioned WDL value path, Mish gain,
truncated-normal initialization, fixed activation scaling and one final masked
BatchNorm in `python/etazero/network.py` are adapted from KataGo
`python/katago/train/model_pytorch.py`. The starting dimensions and global-block
placement follow `b5c192nbt` in `modelconfigs.py`, with the `-fson-mish` variants.
EtaZero derives other widths from trunk channels with alignment to eight and
extends the b5 global-block cadence to configurable depth. It retains its own
Gomoku inputs, six policy outputs and actual-draw WDL, and omits Go pass/score
heads, RepVGG convolutions and dual-head training. Final normalization retains
KataGo's EMA of population mean/std (epsilon 1e-4, momentum 0.001); reciprocal
then multiplication implements division with a cacheable inference operation.
No external source checkout is needed to build or run this network.

`cpp/src/search/search.cpp` adapts fixed KataGo `search.cpp`, `searchnode.h`
and `searchupdatehelpers.cpp` for shared WDL nodes and independent parent edges.
LCB scales weight-square linearly by edge/child visits, while parent aggregation
scales the original weight-square quadratically. Gomoku graph identity stores
exact board/rule/player/input conditions; Go `graphhash.cpp`'s repetition-history
fold is inapplicable to monotonic NOVC play. `tests/reference/check_katago_graph.py`
executes selected unchanged source bodies with scalar shims for independent
validation. Normal builds and runs do not read the reference repositories.

Uncertainty sample/terminal weights and noise-pruned value aggregation adapt
`searchupdatehelpers.cpp` with pure W-L utility (Go score derivative is zero).
Optimistic prior mixing uses ordinary/short optimistic logits and the float
statement from `neuralnet/cudaandrocmbackend.inc`, before temperature/softmax.
`tests/reference/check_katago_search_corrections.py` executes the fixed source bodies
for independent scalar comparisons. Cache keys use exact optimism bytes, whereas
KataGo's NN hash discretizes positive optimism to 1/1024; cache hit rates, RNG
sequences, scheduling and throughput are not claimed equivalent.

PDA budget/input and side/reanalysis production in `sampling.cpp`,
`search_limits.cpp`, `record.cpp` and `game.h` adapt KataGo `program/play.cpp`,
`neuralnet/nninputs.cpp`, `search/searchnnhelpers.cpp` and `trainingwrite.cpp`.
Gomoku has no handicap or komi compensation; finite budget limits follow the
source factors/minimum, while EtaZero int32-max playouts represent no extra limit.
Side response selection restores LCB, recursive ordinary-policy evaluation uses
temperature 1, and both side PDA globals are zero. Outcome-disable reanalysis
suppresses successor policy; main WDL/TD still use the completed game.
`tests/reference/check_katago_sampling.py` executes unchanged source scalar bodies.
Hint/PCR priority, six-turn cheap-probability gates, root hint prior/forced
exploration, hinted-value refresh and early/game/hint forks also follow the
fixed `play.cpp`, `searchhelpers.cpp` and `searchexplorehelpers.cpp`. The NOVC
adapter uses replayable Gomoku text hints (sampling weight only, per-game
training validity 1), exact position comparison and pure W-L candidate ranking
rather than SGF/Go score/komi/seki. Actual candidates are bounded by available
empty cells; terminal starts are rejected. Shared forks persist across requests
and evaluator release within a worker; process restart rebuilds them.
`tests/reference/check_katago_forks.py` compiles the unchanged complete source search
limits body with scalar adapters, covering 3072 hint/PCR/reduced/PDA branches.
Source float target weights and EtaZero's double internal frequencies have a
6e-8 absolute comparison tolerance; disk values are float32. Integer caps,
flags and probabilities retain 1e-12 relative/1e-14 absolute tolerance. Source
RNG and concurrent scheduling equivalence are not claimed.

`tests/reference/check_katago_sampling_weights.py` links the actual production core
library and compares 336 cases to the unchanged source redistribution block
and value-surprise KL. Source float versus double intermediates are compared
with 3e-7 relative / 6e-8 absolute tolerance; the observed maximum absolute
difference is 5.960464477539063e-8 after float32 serialization.

The plain convolutional network in `python/etazero/network.py` follows the fixed
KataGo `b10c128-fson-mish` preset: ten regular residual blocks, global pooling
in blocks five/eight, fson normalization, Mish and v15 heads. The existing NBT
uses `b5c192nbt-fson-mish`. `tests/reference/check_katago_network.py` runs the original
source models with mapped weights to check forward values, gradients, parameter
roles and initialization; input projections adapted to five spatial/six global
Gomoku features, omitted pass/score outputs and
W/L/no-result to W/D/L row selection are explicit Gomoku adaptations. The checker
imports the reference only during development; production has no dependency on
its source directory. This adaptation retains the KataGo MIT license.

The Transformer in `python/etazero/transformer.py` adapts the bare KataGo
`b5c192h3nbttfrs` v17 preset: five NBT2 blocks, fixup/ReLU outer projections,
RMSNorm, three 32-dimensional attention heads, fixed 2D RoPE and SwiGLU with
256 hidden features. Its heads have 32 channels and 64 hidden value units,
without a final BatchNorm. Optional v17 Q retains pure W-L/node-visit targets, stochastic per-output-row int16 quantization and the source BCE weighting. Go score Q is omitted. The
NCHW/SDPA path is a public source alternative to its default NHWC/flex path;
this is not a claim of source execution or throughput equivalence.

`python/etazero/fused_swiglu.py` copies KataGo's MIT-licensed
`python/katago/train/fused_swiglu.py` at the registered commit. Only attribution
and the custom-op namespace (`katago::` to `etazero::`) differ. CUDA training
FP16/BF16 retains its fused forward rounding and recomputed backward. Export
uses serializable SiLU/linear operations with explicit intermediate rounding.
`optimization.py` and `tests/reference/check_reference_formulas.py` extend the registered
source `trainloop_helpers.py` fixup rules, including the separate attention
weight-decay factor. Source files are pinned in `reference_sources.json`; none
of the reference checkouts are needed for production.

Pure W-L Q extraction and training adapt the registered KataGo
`cpp/program/play.cpp`, `cpp/search/searchresults.cpp`,
`cpp/dataio/trainingwrite.cpp`, `python/katago/train/model_pytorch.py`, and
`python/katago/train/metrics_pytorch.py`. Q counts child nodes, including terminal
children; side/reanalysis use their own search. Each repeated writer row receives
independent stochastic quantization. EtaZero uses separate per-game seed streams
and excludes score targets and Go Rand sequencing. `tests/reference/check_katago_q.py`
uses the pinned unmodified source quantizer against the actual core library.
