# Origin of this folder

Unmodified copy of `claude_challenge_system.zip` and `Claude_Challenge_Commands_and_Prompts.md`, sent to the user on 2026-10-05
to help upgrade the Axo skill. The user declared it more trusted than the Axo skill: where the two disagree, follow this folder.
The skill files `compliance-firewall.md`, `research-gate.md`, `postmortem.md` and `commands.md` are condensed, Axo-adapted
versions of it; read the originals here when a detail matters. Not installed as a separate skill, and its global `CLAUDE.md`
and `loop.md` were not installed (they would apply to every project); the three agent prompts are encoded as roles in the
skill's reference files and can be run as subagents when the user asks.

One pack rule is deliberately not adopted (user decision, 2026-10-05): the pack assumes the user trains on Kaggle and sends logs back ("do not train locally"). The user never does that; the agent is the primary runner and trains, validates, checks and packages on the user's PC.
