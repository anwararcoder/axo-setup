# Claude Code Challenge System

This pack is designed for a score-first ML challenge workflow:

SOLVE -> SCORE -> IMPROVE -> FINALIZE -> POSTMORTEM

It deliberately avoids broad research before the first strong solution, because the goal is to preserve the user's current high-performing workflow while adding safety, targeted research, and deterministic finalization.

## Install personal skill and agents

Copy the contents of `_global/` to `~/.claude/`.

Copy:

- `.claude/skills/challenge-engine/` to `~/.claude/skills/challenge-engine/`
- `.claude/agents/` to `~/.claude/agents/`
- `_global/CLAUDE.md` to `~/.claude/CLAUDE.md`
- `_global/loop.md` to `~/.claude/loop.md`

Restart Claude Code after creating a new personal `agents` directory. Skills are available as `/challenge-engine`.

## Recommended usage

### New challenge

Start Claude Code in the challenge workspace and use:

/challenge-engine solve

Then paste the full challenge description and point Claude at the dataset directory, or provide them in the same turn.

### After first Kaggle run

Send the training logs, score, traceback, and any relevant error screenshot/text.

Use:

/challenge-engine improve

Research is intentionally gated here. The main agent should only perform targeted research when the current evidence justifies it.

### Final submission

After selecting the strongest valid solution, use:

/challenge-engine finalize

The goal is a deterministic copy that preserves the winning algorithm rather than simplifying it unnecessarily.

### Postmortem

At the end of a challenge, use:

/challenge-engine postmortem

or explicitly ask the `challenge-postmortem` subagent to update reusable lessons.

## Loop

Do NOT keep a loop running during normal solution development.

The loop is a maintenance mechanism, not model training. Claude Code's `/loop` schedules prompts; it does not change model weights.

For knowledge maintenance, use the bundled `_global/loop.md` as a low-frequency review prompt in a dedicated maintenance session. For example, start a session in your challenge knowledge/workspace and run `/loop 3d` only when you actually want recurring maintenance for that period. Recurring scheduled tasks expire after seven days, so this is not a permanent daemon.

For most challenge work, a manual postmortem once per completed challenge is cheaper and more predictable than keeping a loop alive.

## Optional skill evaluation

Install Anthropic's official skill-creator plugin:

/plugin install skill-creator@claude-plugins-official

Then evaluate the challenge-engine skill with realistic prompts. Keep only changes that improve behavior without materially increasing context or usage cost.
