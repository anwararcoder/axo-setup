# Which Plane Is This? Matching Monthly Flight Logs to Aircraft

- Domain: From Scratch
- Difficulty: Medium
- Scoring: ↑ Higher is better
- Compute: CPU
- Tags: feature-engineering

---

Overview

This is a from-scratch challenge: every model must be trained from zero on the provided data, on CPU, within the time limit. No pretrained weights of any kind and no models trained elsewhere.

For many airlines every scheduled flight is recorded together with the aircraft that flew it. Put together, a month of those records is the diary of one airplane: where it went, how far, how long it stayed in the air and on the taxiway, how many times a day it flew, at what hours, how late it was, when it was cancelled or diverted.

Each test episode holds eight aircraft of the same airline in the same month. For each aircraft you get its month of flying: a sequence of flight records. Separately, and shuffled, you get the eight aircraft as they are described: manufacturer and model, year of manufacture, year it became airworthy, engine manufacturer and model, number of engines, seats, weight class, engine type and thrust.

Match each month of flying to its aircraft: a one-to-one assignment inside every episode.

Evaluation

For each test month of flying, the submitted aircraft is compared with the aircraft that flew it.

```
accuracy = share of months matched to their own aircraft
score    = max(0, (accuracy - 1/8) / (1 - 1/8))
```

A random matching inside each episode is right once in eight and scores about 0; a perfect matching scores 1. The score lies between 0 and 1, higher is better.

Dataset

Files under ./dataset/public/:

* train\_flights.csv: one row per flight of a training month. Columns log\_id, episode, airline, day, day\_of\_week, origin, dest, distance\_mi, sched\_dep, dep, sched\_arr, arr, air\_time\_min, taxi\_out\_min, taxi\_in\_min, sched\_block\_min, block\_min, dep\_delay\_min, arr\_delay\_min, cancelled, diverted.
* train\_aircraft.csv: one row per training aircraft. Columns aircraft\_id, episode, airline, manufacturer, model, year\_mfr, airworthiness\_year, engine\_manufacturer, engine\_model, engines, seats, weight\_class, engine\_type, thrust.
* train\_pairs.csv: episode, log\_id, aircraft\_id for every training month.
* test\_flights.csv, test\_aircraft.csv: the same columns for the test episodes.
* sample\_submission.csv: the required format.

Column meanings:

* log\_id and aircraft\_id are opaque identifiers; the identifier of a month of flying says nothing about the identifier of its aircraft.
* episode identifies the eight aircraft that belong together. Every episode has exactly eight months of flying and eight aircraft, and every month of an episode belongs to one of its aircraft.
* airline is the reporting airline's code, the same for the whole episode.
* day is the day of the month and day\_of\_week runs from 1 (Monday) to 7 (Sunday). origin and dest are airport codes; distance\_mi is the great-circle distance in miles.
* sched\_dep, dep, sched\_arr and arr are local clock times as hhmm; the columns ending in min are minutes; delays are negative for early departures or arrivals; cancelled and diverted are 1 for cancelled or diverted flights; for those flights the actual times, durations and delays can be partly or entirely empty. Any flight value can be empty when not reported.
* manufacturer, model, engine\_manufacturer and engine\_model are as recorded; year\_mfr and airworthiness\_year are years; engines and seats are counts; weight\_class and engine\_type are categorical codes; thrust is the rated engine thrust in pounds, 0 or empty when not recorded. Any other aircraft value, year\_mfr included, can be empty when not recorded.

Compute

Solutions run on the CPU tier and must complete within 90 minutes, training and inference included.

What not to use

Do not look up the flights or the aircraft outside the provided files: no flight-tracking service, fleet list or any other source, and no use of knowledge of real airline fleets or aircraft to identify them. No external data of any kind may be used. Recovering a match by identifying the aircraft is not a solution to this task and is not permitted. Do not infer matches from identifiers, file order or row order. Each episode is matched on its own. Using one test episode to adapt the model for another is not permitted.

Submission

Submit submission.csv with exactly two columns:

* log\_id: string, the test month identifier
* aircraft\_id: string, the identifier of the aircraft you match to it

Requirements:

* One row per log\_id in test\_flights.csv, no duplicate log\_id.
* Every log\_id must be f followed by 12 hexadecimal digits, and every aircraft\_id a followed by 12 hexadecimal digits.
* The matching must be one-to-one: an aircraft\_id given to more than one month counts as a wrong match for every month it is given to.
* Submissions with missing, duplicate or malformed identifiers, or with missing or extra columns, are rejected. An aircraft\_id that is not the aircraft of that month counts as a wrong match.
