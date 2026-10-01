# Reference code and attribution

KataGo is the primary reference for game-thread scheduling, shared batch inference,
bounded queues, background data writing, and selfplay → shuffle → train → export file
boundaries. The checked source is commit `d91ea855110dae533f0aada947b2b7d78cc8a4e1`.
The power-law window function in `python/etazero/shuffle.py` is adapted from KataGo's
`python/shuffle.py`; the two-phase shuffle and queue services follow its architecture.
KataGo's copyright and MIT license are retained in [licenses/KataGo.txt](licenses/KataGo.txt).

The recursive Renju analyzer in `cpp/src/game/rules.cpp` is adapted from the user's
MuZero_V2 working tree, preserving exact-five priority, overline, distinct-four and
recursive live-three semantics. MuZero_V2 is also a reference for the TorchScript /
LibTorch boundary. Its additional forbidden-point observation planes are not part of
EtaZero's six-plane contract.

[reference_sources.json](reference_sources.json) records the actual inspected source
paths and SHA-256 values. These directories are development references; builds and
runs use only files and dependencies inside EtaZero and the selected Python environment.

The reference scope is the stated runtime/data architecture and Renju semantics.
Go rules, KataGo's search/learning enhancements, distributed training services and
its multiple native inference backends are outside the first-round implementation.
