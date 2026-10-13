# Designing the Vertical Profile of Instrument Approaches - Altitude Coding from Lateral Path and Terrain

- Domain: Sequence To Sequence
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: feature-engineering

---

Designing the Vertical Profile of Instrument Approaches - Altitude Coding from Lateral Path and Terrain

Overview

This is a sequence-to-sequence task. The input is the leg sequence of an instrument approach procedure. The output is a sequence of the same length: the altitude constraint the procedure's designer coded on each leg.

An instrument approach is designed in two layers. The **lateral** design is the path: a chain of legs from the initial approach fixes, through the intermediate and final approach fixes, to the missed-approach point at the runway, and then the missed-approach climb and hold. The **vertical** design is the altitude coded on each leg: "at or above 3000 ft" at the intermediate fix, a glide-slope crossing altitude at the final approach fix, "at 1055 ft" over the runway threshold. Each row gives the lateral design, the airport and threshold elevations, the minimum safe sector altitudes and a terrain raster. The task is to write the vertical design leg by leg, exactly as coded.

What the target is, and what varies:

* For every leg, an altitude **descriptor** (at or above, at, at or below, between, or one of the glide-slope and vertical-path forms) and one or two **altitudes** in feet above mean sea level, or no constraint at all.
* Designers set altitudes to clear terrain and obstacles by a margin that depends on the segment, then round them, align them with the glide path and the sector altitudes, and reuse them across legs. The same lateral pattern carries different altitudes at different airports.
* A procedure has 12.0 legs on average (up to 46). In the training data 85.6% of legs carry an altitude and 5.1% carry a second one. The median altitude is 1990 ft above the threshold, and one in ten is more than 3990 ft above it.
* The split is by airport. No airport in the test set appears in training.

Dataset

Files in `public/`:

* `train.csv` (7,105 procedures at 2,032 airports) and `test.csv` (2,373 procedures at 698 airports). Same columns; `train.csv` adds the target `vertical`.
* `train_terrain.npz` and `test_terrain.npz`: arrays `id` [n] and `terrain` [n, 64, 64] (int16, feet), one raster per procedure, in the same row order as the matching CSV.
* `sample_submission.csv`: the required format (every leg without a constraint; it scores 0).

Columns:

* `id`: opaque procedure id.
* `site`: an opaque airport label. Rows with the same `site` are procedures at the same airport.
* `approach_type`: one letter, the coding standard's approach route type. R area navigation, L localizer, I instrument landing system (with glide slope), H required navigation performance, S VOR approach flown on a VOR/DME or VORTAC, V VOR, D VOR/DME, N NDB, B localizer back course, X localizer-type directional aid, P GPS. In the training data: R 63.2%, L 13.4%, I 12.2%, H 4.6%, S 2.4%, V 2.4%, D 0.9%, B 0.3%, N 0.3%, X 0.2%, P 0.1%.
* `elevations`: text, `airport <ft> threshold <ft>`, for example `airport 730 threshold 720`. These are the airport elevation and the landing-threshold elevation, rounded to 10 ft. The threshold is `.` for a procedure not aligned with a runway (a circling procedure, 2.7% of training rows).
* `msa`: the minimum safe altitude sectors published for the airport, in text. Blocks are separated by `;` (one block per sector centre), and the sectors in a block by `/`. Each sector is `from-to:altitude:radius`. The bearings are the published sector boundary bearings, re-expressed in degrees relative to the final approach course; identical bearings mean one full circle. Altitude is in feet and radius in NM. Example: `49-139:3100:25/139-49:2600:25`. `.` means none published.
* `legs`: the procedure's legs in coded order, separated by `|` . Each leg has 10 space-separated fields, with `.` for an empty field:

  1. segment: `T1`, `T2`, ... for the transitions (feeder routes into the approach), `F` for the final approach route, `M` for the missed approach (from its first leg).
  2. fix: a token for the fix the leg ends at `P1`, `P2`, ... in order of first appearance; the same token within a row is the same fix; `RWY` is the runway threshold; `-` for a leg that ends at no fix, such as a climb to an altitude).
  3. fix kind: `WPT` waypoint, `VOR`, `NDB`, `RWY`, `UNK` (position unknown), or `-`.
  4. waypoint description code, 4 characters, `_` for blank.

     + Character 1: `E` essential waypoint, `G` runway, `V` VOR used as a waypoint, `N` NDB used as a waypoint.
     + Character 2: `E` end of a continuous segment, `Y` fly-over fix; `B` is a rare flag (14 training legs).
     + Character 3: `M` first leg of the missed approach, `S` named step-down fix; `R` is a rare flag (59 training legs).
     + Character 4 (the fix's role): `A` initial approach fix, `B` intermediate fix, `I` final approach course fix, `F` final approach fix, `M` missed-approach point; `D` is a rare combined role (21 training legs).
  5. path/terminator (the leg type), for example `IF` initial fix, `TF` track to fix, `CF` course to fix, `DF` direct to fix, `RF` constant-radius arc, `AF` DME arc, `CA` course to an altitude, `VAVIVM` heading legs, `HMHFHA` holds.
  6. turn direction `L` or `R`.
  7. course, whole degrees from 0 to 359, relative to the final approach course.
  8. leg distance in NM, or `T` plus minutes for a timed leg.
  9. `x`: NM from the reference point to the fix along the final approach course, positive on the approach side (before the runway).
  10. `y`: NM to the right of the inbound final approach course.

The reference point is the runway threshold when `elevations` has a threshold, and the airport reference point otherwise. Fields 9 and 10 are `.` for a leg without a fix or whose fix position is unknown `UNK`).

Terrain: the raster is in each procedure's own frame. Cell `[i, j]` covers `x` in [i - 16, i - 15) NM and `y` in [j - 32, j - 31) NM, so the reference point is in cell `[16, 32]`. The value is the highest ground elevation in that 1 NM cell, in feet, with water at 0. The raster shows the ground, not towers, buildings or trees.

Target

`vertical` (train only): one token triple per leg, in the order of `legs`, joined by `|` .

* Descriptor:

  + `+` at or above altitude 1; `-` at or below; `@` at.
  + `B` between, with altitude 1 the upper and altitude 2 the lower bound; `C` at or above altitude 2.
  + `G`, `H`, `I`, `J` glide-slope forms, where altitude 2 is the glide-slope altitude: `H` and `J` are at or above altitude 1, and `I` and `J` mark the glide-slope intercept.
  + `V` step-down fix, with a vertical-path altitude in altitude 2; `X` and `Y` other vertical-path forms.
  + `.` no constraint.
* Altitude 1 and altitude 2: integer feet above mean sea level, `.` when empty. A leg without a constraint is `. . .`.

Counts in the training data: `+` 60,334, `.` 12,288, `@` 8,479, `V` 2,486, `H` 863, `J` 854, `B` 103, `-` 60, `I` 15, `G` 2, `X` 1 legs.

Example for a five-leg procedure: `. . . | + 3000 . | J 3000 1700 | H 1700 1700 | @ 1055 .`

Evaluation

score = 0.50 x closeness + 0.30 x exact + 0.20 x descriptor

* **closeness**: averaged over every altitude field that the truth or the prediction fills (altitude 1 and altitude 2 of every leg) across all test procedures. Each field scores 1 - |predicted - true| / 500 ft, floored at 0. A field filled on one side only scores 0, so a missing altitude and an altitude written where the designer set none cost the same.
* **exact**: the share of those same fields predicted within 50 ft, inclusive. A one-sided field is a miss.
* **descriptor**: Cohen's kappa between predicted and true descriptors over every leg of every test procedure `.` counts as a class), floored at 0. Predicting the most common descriptor everywhere scores 0.

Edge cases:

* Writing an altitude where the truth has none, including a second altitude on a leg that has only one, lowers closeness and exact.
* A leg whose descriptor is `.` has no altitudes; any altitudes written on it are ignored.
* Descriptors are case-insensitive.
* A malformed leg (not three fields, an unknown descriptor, or an altitude that is neither `.` nor a finite number) counts as an empty prediction for that leg: no descriptor and no altitudes. A submission with an unknown descriptor on every leg therefore scores 0.
* Legs are matched by position. A row with fewer legs than its test row leaves the rest empty; extra legs are ignored.
* The answer key scores exactly 1.0.

Submission

A CSV with exactly two columns, `id` and `vertical`, and one row per test id.

Requirements:

* Every test id appears once. A missing id, a duplicate id, an id that is not a test id, or any other column makes the submission invalid.
* `vertical` follows the target format above, with the legs in the order of that row's `legs` in `test.csv`.
* Compute tier: CPU (10 cores, 62.5 GiB RAM).

What Not To Use

* Any published instrument procedure database, chart, navigation-data product, airport directory or flight-planning source, or any copy or mirror of one. Do not look up or match rows against them.
* Any attempt to identify the airport, runway, procedure or fixes behind a row, or to undo the anonymous fix tokens, frame or site labels.
* Asking a pretrained language model to recall a procedure's altitudes. A pretrained model used only to compute features is fine.
* Row order and the `id` values as predictive signal.
* Any data beyond the files in `public/`.

A solution that uses any of these is invalid, whatever it scores.
