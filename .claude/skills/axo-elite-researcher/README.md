# Axo Elite Researcher

A research-first workflow for the challenges that `fetch_challenges.py` puts in `challenges/`. Claude scans,
ranks, researches, builds and checks; you decide and you upload. Claude never submits or spends a credit.

| You want to | Type |
|---|---|
| see every challenge and what this PC can do | `/axo-elite-researcher scan` |
| pick a challenge | `/axo-elite-researcher select` (paste fresh status blocks if you have them) |
| research the chosen challenge | `/axo-elite-researcher research <slug>` |
| build and validate a solution | `/axo-elite-researcher solve <slug>` |
| react to a score or an error | `/axo-elite-researcher improve`, then paste the whole bot message |
| check a folder before uploading | `/axo-elite-researcher audit` |
| get the final upload folder | `/axo-elite-researcher finalize` |
| know where things stand | `/axo-elite-researcher status` |

Upload only a `challenges/<slug>/submission_vN/` folder that Claude has called ready, following its
`HOW_TO_SUBMIT.md`.

## Files

- `SKILL.md`: the operations and the rules Claude follows.
- `references/`: ranking and model policy, research protocol, experiments and the score loop, audit and
  finalization, and the knowledge reused from the `axo` skill.
- `scripts/hw_profile.py`: read-only machine profile.
- `scripts/scan_challenges.py`: read-only challenge inventory; never prints dataset links.
- `scripts/guarded_run.py`: runs one experiment under a memory cap, so it cannot freeze the PC, and logs it.
- `assets/`: templates for `RESEARCH.md` and `LOG.md`.

All three scripts use only the Python standard library: `python3 <script> --help`.
