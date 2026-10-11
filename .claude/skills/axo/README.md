# Axo skill for Claude

It takes you from "which challenge?" to a ready-to-upload folder.
You talk. Claude does the work. You upload.

---

## 1. Install (2 minutes)

**Claude Code (recommended, because Claude trains on your PC):**

1. Unzip `axo.zip`.
2. Move the `axo` folder here:
   - Windows: `C:\Users\<you>\.claude\skills\`
   - Mac / Linux: `~/.claude/skills/`
3. Check: this file must exist → `.claude/skills/axo/SKILL.md`
4. Restart Claude Code.

**Claude app:** Settings → Capabilities → Skills → upload `axo.zip`.

**Test it:** type `/axo`. If Claude answers about Axo challenges, you are done.

---

## 2. Pick a challenge

Do only this:

1. Copy **every open challenge post** from the Telegram group.
2. For each one, copy its **status block** from the bot too.
3. Paste it all into Claude with this line:

```
/axo select — which one should I do?
```

4. Claude answers with **one pick** and why.
5. Say **"go"**, or say which one you want instead.

That's it. Do not compare them yourself.

**In a hurry? Check 3 things:**

| Look at | Good sign | Skip it |
|---|---|---|
| Compute | `CPU` | Needs a big pretrained model |
| Data | Tables, text, small files | GBs of images, video or audio |
| Seats | Few solvers so far | 8/10 or more already in |

Small and fast beats interesting. You want a score in 30 minutes, not in 3 hours.

---

## 3. Solve it

One step at a time. Do not skip ahead.

**Step 1. Start.** Put the challenge markdown and the dataset in one empty folder. Open Claude there. Type:

```
/axo solve
```

**Step 2. Upload the first version.** Claude gives you a `submission/` folder in about 20 minutes. Open `HOW_TO_SUBMIT.md` inside it. It tells you which file goes where. Upload.

> The first version is meant to be weak. It is a test run. Upload it anyway.

**Step 3. Paste the score back.** Copy the bot's "Submission scored" message into Claude. Nothing else needed.

**Step 4. Repeat.** Claude builds a better version. You upload. You paste the score. Loop.

**Step 5. Finish.** When you are happy with the score:

```
/axo finalize
```

Upload that folder. Done.

---

## 4. What to type, when

| What happened | Type this |
|---|---|
| I need to choose a challenge | `/axo select` + the posts |
| I chose one | `/axo solve` |
| I got a score | paste the bot message |
| I got an error | paste the error |
| I want a better score | `continue improving` |
| Time is almost up | `now` |
| I am happy with the score | `/axo finalize` |
| Is it safe to submit? | `/axo audit` |
| The challenge closed | `/axo postmortem` |

Lost? Type: **`where are we and what do I do next?`**

---

## 5. Four rules

1. **Paste, don't explain.** Bot messages, errors, scores: paste them as they are.
2. **You have 6 uploads per challenge.** Do not spend one on a guess.
3. **Only upload what is in `submission/`.** Never put files together by hand.
4. **The bot is the boss.** If the bot says something different from Claude, tell Claude.

---

## Good to know

- You need Python on your PC. A GPU helps, but is not required for CPU challenges.
- The first time, Claude checks your machine by itself. Let it.
- The notes inside the skill come from earlier challenges on one specific PC. Your timings may differ.
