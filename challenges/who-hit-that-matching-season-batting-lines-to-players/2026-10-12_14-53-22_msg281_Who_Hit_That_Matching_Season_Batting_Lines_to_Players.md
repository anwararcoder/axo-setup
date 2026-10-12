# Who Hit That? Matching Season Batting Lines to Players

- Domain: Other
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: feature-engineering, small-data

---

Overview

Each test episode holds eight regular position players of one Major League Baseball team in one season. For each player you get his season batting line for that team: how often he came to the plate and what happened (singles, doubles, triples, home runs, walks, strikeouts, hit by pitch, sacrifices, reaching on errors, double plays grounded into), the runs he scored and drove in, his steals and times caught stealing, the same outcomes split by the throwing hand of the pitcher he faced, and the games he started at each fielding position. Separately, and shuffled, you get the eight players themselves: age, height, weight, batting side, throwing hand, years since his major-league debut, and whether he was born in the United States. The team and the season are not given.

Match each batting line to its player.

What makes this hard

The eight players of an episode played for the same team in the same season, so they share a ballpark, a league and an era. A batting line describes what happened over a season, which depends on the player's role and luck as much as on who he is; a player's description says nothing about how he played. Most players are right-handed, of similar build and in their late twenties. The seasons span five decades in which the game itself changed, and the season of an episode is not given.

Evaluation

For each test batting line, the submitted player is compared with the player who produced it.

```
accuracy = share of lines matched to their own player
score    = max(0, (accuracy - 1/8) / (1 - 1/8))
```

A random matching inside each episode is right once in eight and scores about 0; a perfect matching scores 1. The score lies between 0 and 1, higher is better.

Dataset

Files under ./dataset/public/:

* train\_lines.csv: one row per training batting line. Columns line\_id, episode, pa, singles, doubles, triples, home\_runs, walks, intentional\_walks, strikeouts, hit\_by\_pitch, sacrifice\_bunts, sacrifice\_flies, reached\_on\_error, grounded\_into\_dp, runs, runs\_batted\_in, stolen\_bases, caught\_stealing, pa\_vs\_lhp, on\_base\_vs\_lhp, total\_bases\_vs\_lhp, strikeouts\_vs\_lhp, pa\_vs\_rhp, on\_base\_vs\_rhp, total\_bases\_vs\_rhp, strikeouts\_vs\_rhp, games\_started, starts\_c, starts\_1b, starts\_2b, starts\_3b, starts\_ss, starts\_lf, starts\_cf, starts\_rf, starts\_dh.
* train\_players.csv: one row per training player. Columns player\_id, episode, age, height\_in, weight\_lb, bats, throws, years\_since\_debut, born\_usa.
* train\_pairs.csv: episode, line\_id, player\_id for every training line.
* test\_lines.csv, test\_players.csv: the same columns for the test episodes.
* sample\_submission.csv: the required format.

Column meanings:

* line\_id and player\_id are opaque identifiers; the identifier of a line says nothing about the identifier of its player.
* episode identifies the eight players that belong together. Every episode has exactly eight lines and eight players, and every line of an episode belongs to one of its players.
* pa counts plate appearances; the outcome columns count the plate appearances that ended that way. on\_base\_vs\_lhp counts hits, walks and hit-by-pitches against left-handed pitchers, total\_bases\_vs\_lhp the bases from hits against them; the *vs*rhp columns are the same against right-handed pitchers.
* runs, runs\_batted\_in, stolen\_bases and caught\_stealing are counted from the play-by-play record of every game and can differ slightly from official totals.
* starts\_c … starts\_dh count the games started at catcher, first base, second base, third base, shortstop, left field, centre field, right field and designated hitter.
* age is the season year minus the player's birth year; height\_in is in inches and weight\_lb in pounds, as listed for the player's career; bats is L, R or B (both sides); throws is L or R; born\_usa is 1 for players born in the United States.

Only players with at least 150 plate appearances for the team whose most frequent starting position was not pitcher are included. Episodes of the same team and season are on the same side of the split.

Compute

Solutions run on 10 CPU cores and must complete within 90 minutes, training and inference included. Public, general-purpose pretrained weights are allowed. No external data of any kind may be used.

What not to use

Do not look up the lines or the players outside the provided files: no search of baseball statistics databases or any other source, and no use of knowledge of real players, teams or seasons to identify them. Recovering a match by identifying the player is not a solution to this task and is not permitted. Do not infer matches from identifiers, file order or row order. Each episode is matched on its own. Using one test episode to adapt the model for another is not permitted.

Submission

Submit submission.csv with exactly two columns:

* line\_id: string, the test line identifier
* player\_id: string, the identifier of the player you match to it

Requirements:

* One row per line in test\_lines.csv, no duplicate line\_id.
* Every line\_id must be l followed by 12 hexadecimal digits, and every player\_id p followed by 12 hexadecimal digits.
* Submissions with missing, duplicate or malformed identifiers, or with missing or extra columns, are rejected. A player\_id that is not the player of that line counts as a wrong match.
