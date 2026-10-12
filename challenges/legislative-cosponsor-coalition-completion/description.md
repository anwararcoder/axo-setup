# Legislative Cosponsor Coalition Completion

- Domain: Recommendation
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: text

---

**Overview:**  
When a bill is introduced in a legislature, it already has a small coalition behind it: the member who sponsors it and the original cosponsors who signed on the same day, usually after helping to prepare it. Over the following weeks and months other members join as cosponsors, persuaded by the bill's subject, by the sponsor, by colleagues from their region or party, or by groups outside the chamber. Offices that build support for a bill keep asking the same question: given this bill and the members already behind it, who should we approach next?

This challenge turns that question into a recommendation task. Each case is one bill introduced in the lower chamber of a national legislature during one of three consecutive two-year terms. You are given the bill's title, policy area, subject terms, committees and an official plain-language summary, its sponsor and its original cosponsors. Recommend the members who joined it later, as a ranked list of members of the chamber.

Task definition:  
The prediction unit is one bill. For each evaluation bill you return one ranked list of member ids, best first. The candidates are the members listed in members.csv for the bill's term, other than its sponsor and original cosponsors; there are roughly 450 members per term. A list may hold at most 100 members, and only its first 100 entries are scored.

What the inputs are:  
Members are identified by opaque codes (m0001, m0002, ...) that stay the same across the three terms, so a member's history in one term is that member's history in another. members.csv gives, for each term, every member who appears in the data, with their party and their region as opaque codes: party\_1, party\_2, ... for the party and region\_01, region\_02, ... for the electoral region they represent. Members with the same region code represent the same region, and members with the same party code belong to the same party. Every bill has at least one original cosponsor and at least three later cosponsors, and at most 60 later cosponsors. Later cosponsors who withdrew are not counted.

Supervision and split:  
train\_labels.csv gives the later cosponsors of every training bill. The split is by bill: evaluation bills come from all three terms, and every evaluation bill has a title that, ignoring years, is shared with no other bill in the data, so bills re-introduced under the same title are not evaluation bills. No sponsor has more than four evaluation bills. Case ids are random codes. There are 5,442 training bills and 1,000 evaluation bills.

**Evaluation:**  
The metric is mean average precision at 100, where HIGHER IS BETTER.

For one bill, let L be the set of members who joined it later, and read the submitted list after removing repeated ids. Then

AP@100 = (1 / size of L) x sum over positions k = 1..100 at which the list holds a member of L of (number of members of L among the first k entries) / k

The final score is the mean of AP@100 over all evaluation bills. A submission must contain every evaluation bill; a file that leaves any evaluation bill out is rejected. A bill whose list is empty scores 0. Ids that are not members of the bill's term simply occupy a position.

Why this metric: a recommender for coalition building produces a ranked list of members to approach, and the useful lists are those that put the members who actually joined near the top. Average precision rewards exactly that: every later cosponsor contributes according to how high it is ranked and how many wrong members are ranked above it, and it depends only on the order of the list. The cut-off at 100 keeps the list short relative to the chamber while always leaving room for every later cosponsor, since no bill has more than 60; for the same reason the denominator is the size of L for every bill, so a perfect list scores exactly 1 whatever the size of the coalition. Taking the mean over bills gives every bill the same weight, so bills with very large coalitions do not dominate the score.

Floor and ceiling: a list that starts with all of a bill's later cosponsors scores 1 for that bill, and a list that holds none of them in its first 100 entries scores 0. The score therefore runs from 0 to 1, and a list in random order scores close to 0.

**Dataset:**  
The following files are provided in public/.

train.csv - one row per training bill, 5,442 rows.  
case\_id - text - opaque identifier of the bill.  
term - text - the term the bill was introduced in: t1, t2 or t3, in time order.  
title - text - the bill's title.  
policy\_area - text - the single policy area assigned to the bill (unassigned if none).  
subjects - text - a JSON list of the subject terms assigned to the bill.  
committees - text - a JSON list of the committees the bill was referred to.  
summary - text - the official plain-language summary of the bill as introduced.  
sponsor - text - member id of the sponsor.  
original\_cosponsors - text - member ids of the original cosponsors, separated by single spaces.  
test.csv - one row per evaluation bill, 1,000 rows, with exactly the same columns as train.csv in the same order.  
train\_labels.csv - the later cosponsors of every training bill. Join on case\_id.  
case\_id - text - identifier matching train.csv.  
later\_cosponsors - text - member ids of the later cosponsors, separated by single spaces.  
members.csv - one row per member and term.  
member\_id - text - opaque member code, the same in every term.  
term - text - t1, t2 or t3.  
party - text - opaque party code.  
region - text - opaque code of the region the member represents.  
sample\_submission.csv - a correctly formatted submission listing 100 eligible members of each bill's term in an arbitrary fixed order.  
case\_id - text - identifier matching test.csv.  
ranked\_members - text - the ranked list.

Submission:  
Submit a CSV with these two columns.

case\_id - text - one row for each case\_id in test.csv, 1,000 rows plus a header row.  
ranked\_members - text - up to 100 member ids, best first, separated by single spaces.

Example, with fabricated ids:

case\_id,ranked\_members  
b0a1b2c3d4e,m0412 m0087 m0301 m0255  
b5f6a7b8c9d,m0019 m0533 m0102

Requirements: a header row and one row per evaluation bill, in any order. Ids may also be separated by commas and are matched case-insensitively. Entries after the 100th are ignored, and an id that repeats an earlier one is ignored. A duplicated case\_id, a missing case\_id or ranked\_members column, a file that leaves out any evaluation bill, or a file that shares no case\_id with the evaluation set is rejected.

**Rules:**  
The only valid input signal is the content of the bill rows, members.csv and the training labels, together with what is learned from the training bills.

The following approaches are not allowed:

* Looking up the source. These bills and their cosponsors are public records. Searching for, downloading or matching against any copy of these bills, their cosponsor lists or any other legislative record is not allowed. The only data is the files in public/.
* Identifying the members. Member, party and region codes are opaque on purpose; mapping them to real people, parties or places, or using any outside information about the members, is not allowed.
* Using case\_id strings or the row order of any file as a prediction signal. They are random codes issued so that files can be joined.
* Producing rankings with hand-written rules. The ranking must come from a model trained on the training bills, for example a learned ranker or a neural recommender. Counts such as how often a member has joined the sponsor's bills, party or region agreement and text similarity may be inputs to that model, but hand-written rules may not decide, filter or override the ranking; a solution that keeps most of its score with the trained model removed is rule-based and not allowed.
* Reasoning across evaluation bills. Each bill is one input. Building member statistics from the evaluation bills' sponsors or original cosponsors, pseudo-labelling, test-time adaptation, self-supervised training on evaluation texts, or letting the ranking of one bill depend on another evaluation bill is not allowed.
* Hardcoding rankings for specific evaluation bills. Every submitted list must be the unedited output of the trained model's inference over that bill; writing or editing lists by hand, or keeping any lookup keyed by case\_id or text, is hand-labelling.
* Using a pretrained language model without training it on the supplied data, for example by prompting it with a bill and the member codes, as the model that produces the rankings.
* Using private, role-gated or API-key-based models, or calling any external inference API.
* Using non-reproducible external weights or artifacts that are not publicly available.

**Pretrained model policy:**  
Publicly available pretrained text models, such as sentence encoders or open-weight language models, may be used as components of the trained recommender to read the bill texts, provided their own weights are fine-tuned on the training bills; freezing some layers is acceptable. Embeddings from a frozen pretrained model fed to a separately trained head do not count as fine-tuning that model. Models or embeddings trained on legislative records of this chamber are outside data and are prohibited, as stated above.
