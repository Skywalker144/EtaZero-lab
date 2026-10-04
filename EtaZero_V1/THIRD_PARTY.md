# Reference code and attribution

[reference_sources.json](reference_sources.json) records the inspected source revisions, paths and SHA-256 values. KataGo is pinned to `d91ea855110dae533f0aada947b2b7d78cc8a4e1`; KataGomo is pinned to `df152116e3787c75c6a3de099d261ca092b7dfc1`. MuZero and SkyZero working-tree references are identified by their registered file hashes. Builds and normal runs use EtaZero code and the selected Python environment; external source checkouts are needed only for optional reference checks.

KataGo copyright and MIT terms are retained in [licenses/KataGo.txt](licenses/KataGo.txt), and KataGomo terms in [licenses/KataGomo.txt](licenses/KataGomo.txt). These notices also cover the corresponding KataGo-derived modules inspected through SkyZero. Runtime semantics are maintained in the algorithm and implementation documents; this file records attribution and adaptation scope.

| Adaptation | Source | EtaZero scope |
|---|---|---|
| Parallel selfplay and inference | KataGo game threads, NN services, caches and synchronization | `cpp/src/selfplay/`, `inference/`, `search/`; shared batching, pooled synchronization, tiered child statistics and reusable worker resources |
| PUCT and search corrections | KataGo search helpers, node statistics and backend logit mixing | WDL-only FPU, forced playouts, policy pruning, LCB, value weighting, uncertainty, optimistic policy and noise pruning; Go score terms excluded |
| Graph search | KataGo `search.cpp`, `searchnode.h`, `searchupdatehelpers.cpp` | Shared WDL nodes, independent parent edges, catch-up visits and root reclamation; Go repetition-history folding excluded for NOVC Gomoku |
| Sampling and supervision | KataGo `play.cpp`, `trainingwrite.cpp`, NN inputs and metrics | PCR, Reduce Visits, surprise weighting, PDA, side/reanalysis and hint/fork adaptations; replayable Gomoku text replaces SGF/Go-specific state |
| Q targets | KataGo child-node extraction, writer quantization and v17 metrics | Pure W−L Q, node visits and per-output-row stochastic int16 quantization; Go score Q excluded |
| Data and replay | KataGo writer, `shuffle.py`, training data generator | Packed observations, file-boundary snapshots, power-law windows, keep-target sampling, scatter/merge and gap-delaying repeat order |
| NBT and plain networks | KataGo `model_pytorch.py`, `modelconfigs.py` | fson/Mish trunks, masked normalization and six policy/WDL heads; Gomoku input projection, width scaling and omitted Go heads are adaptations |
| Transformer | KataGo bare `b5c192h3nbttfrs` v17 preset | Fixup/ReLU NBT2, RMSNorm, RoPE, attention and SwiGLU; public NCHW/SDPA alternative to the source default path |
| Fused SwiGLU | KataGo `python/katago/train/fused_swiglu.py` | `python/etazero/fused_swiglu.py` copies the registered MIT source with attribution and custom-op namespace changes; export uses serializable decomposed operations |
| Optimizer and augmentation | Fixed KataGo fson/fixup training branches; SkyZero V8.1 training and D4 references | Parameter roles, LR/WD scaling, norm metrics, warmup, Lookahead/SWA and batch D4; local consumption clocks and restorable RNG/state are EtaZero adaptations |
| Balanced opening | KataGomo `randomopening.cpp`, `playutils.cpp` | No-VC skeleton, root perspectives, balance weights, retry/fallback and policy init; EtaZero RNG, WDL interface, cancellation and removal of negative sampling epsilon |
| Gomoku inputs | SkyZero V7.19 `gomoku.h`, `selfplay_manager.h`, `nets.py` | Five spatial planes, four initial global features, global projection and per-row forbidden-feature dropout; two additional globals use KataGo PDA semantics |
| Renju rules | User's MuZero_V2 and SkyZero implementations; KataGomo independent rule reference | `cpp/src/game/rules.cpp`; recursive live-three, distinct-four, overline and exact-five semantics; optional checks compile the registered external detector |
| Runtime, scheduling and plotting | MuZero_V2 runtime/layout references; SkyZero V7.19/V8.1 umbrella and GPU scheduling | Complete-round publication, cold-start quota, deterministic random evaluator, weight-only shared initialization and local scheduling; no claim of asynchronous/distributed parity |
| Evaluation and Elo | MuZero_V2 arena/Elo/match; KataGo match profiles and root-visit convention | Paired openings, per-game resume, joint fit and bootstrap; EtaZero model identities, manifests and native search replace the source execution path |
| Web workbench | User's MuZero `web/`, V2 `engine.py` and `eval_main.cpp` | Repository-level `web/` and native `serve`; persistent session protocol, published-model validation and canvas-aware search diagnostics |
| MuZero normalization and gradient boundaries | MuZero_V2 `network.py` | `python/etazero/muzero/network.py`; FP32 masked min/max normalization and gradient scaling |
| MuZero dense residual trunk | MuZero_V2 masked ResNet | `python/etazero/muzero/resnet.py`; masked per-sample normalization, SiLU and two-convolution residual blocks; EtaZero retains its own inputs and heads |
| MuZero sequence training | MuZero_V2 replay and training | `python/etazero/muzero/data.py`, `training.py`; unroll, absorbing-state and gradient boundaries adapted to EtaZero row multiplicity, side masks and optimizer modes |
| Latent inference and search | MuZero_V2 initial/recurrent boundary; EtaZero search math | `cpp/src/muzero/`; independent device-local latent ownership, batching and search; not a claim of complete V2 search parity |

EtaZero's CUDA upload prefetch, authenticated training-view cache, restorable snapshot eviction, persistent worker protocol, exact input cache keys and separate RNG streams are local adaptations. Source RNG sequences, scheduling, throughput and complete repository-level equivalence are not claimed. The source models' pass, score, ownership and no-result outputs have explicit Gomoku mappings or are outside the stated scope.

The hint loader links the environment's OpenSSL Crypto library to hash the bytes it parses. OpenSSL is a build dependency; its cryptographic implementation is not copied into EtaZero.

Optional reference check locations and coverage are listed in [验收与限制](docs/implementation.md#验收与限制). Historical V0 results retain their own source/configuration identities in the repository-level audit records; they are not current V1 acceptance evidence.
