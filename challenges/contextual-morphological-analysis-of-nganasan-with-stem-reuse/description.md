# Contextual Morphological Analysis of Nganasan with Stem Reuse

- Domain: Sequence To Sequence
- Difficulty: Hard
- Scoring: ↑ Higher is better
- Compute: A10G
- Tags: text

---

**Contextual Morphological Analysis of Nganasan with Stem Reuse**

Overview

Build a sequence-to-sequence system that extends a checked Nganasan interlinear analysis to a new utterance. Given an annotated support utterance, the new utterance's ordered words and nearby unannotated context, generate surface morphemes and underlying morphemes for every new word, explicitly identifying stems inherited from the support utterance. Training is optional: you may train from scratch or fine-tune the permitted pretrained checkpoint.

The system assists an editor documenting an endangered language of northern Siberia. Surface forms change under inflection and sound alternation; underlying forms maintain a consistent lexical analysis. A mistaken stem identity or stem realization adds wrong lexical evidence to an edition, while a mistaken suffix usually requires a local correction. Both kinds of evidence must remain linked to the original word order.

Your complete solution must finish within 90 minutes, including software setup, preprocessing, training, validation, inference and submission writing. Use one NVIDIA A10G with 24 GB VRAM and at most 62 GB system RAM.

Each example includes a checked utterance from the same conversation or narrative as the new words. You must decide when its underlying stems can be reused and when a new stem must be generated, while retaining both surface and underlying analyses. The output is an editable inheritance record: writing an existing stem as if it were new loses its connection to the checked evidence. No translations or lexical glosses are supplied or requested.

Dataset

```
dataset/public/

├── train.csv

├── test.csv

└── sample_submission.csv
```

There are 4,000 training rows and 2,000 test rows. All structured text columns contain JSON. A row represents one new utterance; its support analysis can accompany several rows. Related speakers and communications share an opaque `group`. The 22 training groups and five test groups are disjoint. Use whole groups for validation, and include validation-side support analyses as inputs to your validation pipeline.

`train.csv`

```
id,group,anchor,words,context,analysis
```

* `id`: unique opaque example identifier.
* `group`: opaque dependency group for validation and evaluation aggregation.
* `anchor`: ordered array of checked word objects, each with `word`, `surface` and `canonical`.
* `words`: ordered array of word entries in the new utterance; whitespace within an entry does not create additional output positions.
* `context`: object with `before` and `after`, each an array of nearby utterances represented as word arrays; an unavailable side is `[]`.
* `analysis`: the target inheritance record, described below.

For example, a checked word object has this form:

```
{"word":"babiɁ","surface":["babi","Ɂ"],"canonical":["bahi","Ɂ"]}
```

`surface` records the pronounced morphemes in order. `canonical` records their underlying forms in order, beginning with one lexical stem. The two arrays need not have equal lengths: underlying zero morphemes are retained, and surface zero morphemes are omitted. A common underlying zero marker is the private-use character U+F1B9, written `"\uf1b9"` in escaped JSON. Preserve it as supplied; do not replace it with a visually similar character. An empty underlying element `""` is also valid. Suffixes include derivational elements and clitics after the leading stem. Words with multiple lexical stems are outside this task.

Input words and morphemes use lowercase NFC Unicode with terminal punctuation removed from words. Preserve phonological characters, apostrophes, glottal stops, combining marks and vowel length; a plain ASCII approximation changes the analysis. Morpheme strings contain no whitespace.

`test.csv`

```
id,group,anchor,words,context
```

These fields have the same meaning as in training. The new utterance's analysis is to be predicted. Each checked `anchor` remains available at inference time.

`sample_submission.csv`

```
id,analysis
```

The sample uses `[]` to abstain on every utterance. This is a valid representation with score zero.

Submission format

Write `working/submission.csv` with the prediction columns:

```
id,analysis
```

Provide one prediction for every test ID. Rows and named columns may be reordered; additional uniquely named columns are ignored.

`analysis` is an array with one object per new word in the same order. Each object contains `surface`, a nonempty array of nonempty morpheme strings; `suffixes`, an array of underlying morpheme strings after the stem, retaining zero markers and allowing empty elements; and exactly one of these identities:

* `reuse`: a nonnegative integer giving a zero-based word position in this row's checked `anchor`.
* `stem`: a nonempty underlying stem string, used when that stem is absent from the checked `anchor`.

If the true underlying stem equals the first `canonical` element of any checked word, use `reuse`. Any checked position with that same stem is equivalent; training records use the earliest position. Otherwise generate `stem`. Reconstruct a word's canonical analysis by prepending its reused or generated stem to `suffixes`. Identity depends on the canonical stem spelling, not an unprovided semantic sense label.

The empty array is an utterance-level abstention. An incorrect word count or a `reuse` position beyond the checked utterance is a scoreable mistake and gives that utterance zero. Malformed JSON, duplicate keys, inappropriate value types, negative reuse indices, duplicate IDs and missing IDs are invalid submissions. Nonempty morphemes cannot contain whitespace. Grading ignores case and differences between canonically equivalent Unicode encodings.

These two inputs use made-up forms to illustrate the representation. For `example_a`, `anchor` is `[{"word":"demobiɁ","surface":["demobi","Ɂ"],"canonical":["demohi","Ɂ"]}]` and `words` is `["demobintə"]`. Assume its surface analysis is `["demobi","ntə"]` and underlying analysis is `["demohi","ntə"]`. For `example_b`, use the same checked anchor and `words` `["demoŋa"]`; assume its surface analysis is `["demoŋa"]` and its underlying analysis is `["demoŋa","\uf1b9"]`, including a zero suffix. Both examples can use `{"before":[],"after":[]}` as context.

```
id,analysis

example_a,"[{""reuse"":0,""surface"":[""demobi"",""ntə""],""suffixes"":[""ntə""]}]"

example_b,"[{""stem"":""demoŋa"",""surface"":[""demoŋa""],""suffixes"":[""\uf1b9""]}]"
```

The first prediction inherits `demohi` despite its different surface realization. The second generates a new stem and retains the zero suffix without inserting an empty surface morpheme.

Evaluation

The maximizing score ranges from zero to one. A row earns credit only when every canonical leading stem, every leading surface morpheme and every reuse-versus-new decision is correct: incorrect lexical evidence makes the whole proposed extension unsafe to accept. Conditional on that check, credit is the fraction of words with both complete surface morphology and complete underlying suffix morphology correct.

The exact calculation is:

```
def utterance_score(predicted, expected, anchor_stems):

    if len(predicted) != len(expected) or not predicted:

        return 0.0

    for p, t in zip(predicted, expected):

        p_stem = resolve_identity(p, anchor_stems)  # missing position gives None

        t_stem = resolve_identity(t, anchor_stems)

        if p_stem != t_stem:

            return 0.0

        if p["surface"][0] != t["surface"][0]:

            return 0.0

        if ("reuse" in p) != (t_stem in anchor_stems):

            return 0.0

    correct = sum(

        p["surface"] == t["surface"] and p["suffixes"] == t["suffixes"]

        for p, t in zip(predicted, expected)

    )

    return correct / len(expected)

group_scores = [mean(scores_for_group) for group in evaluated_groups]

score = mean(group_scores)
```

Here `anchor_stems` is the first `canonical` element of each checked word. All string comparisons apply NFC normalization and lowercase conversion. Array order and morpheme boundaries matter; an extra, missing or incorrect suffix loses that word's credit. A correct stem with wrong inflection retains the opportunity for partial credit on the remaining words. Groups receive equal weight, so a long recording cannot dominate the score.

Not allowed

* External linguistic datasets, dictionaries, grammars or annotated corpora.
* Pretrained checkpoints other than the permitted revision, including models previously fine-tuned on morphological or interlinear annotation.
* Remote model or prediction APIs.
* Manual annotation of test targets.
* Retrieving annotations for the supplied utterances from an external source.
* Hard-coding predictions from test identifiers or exploiting file ordering, serialization artifacts or evaluation behavior.
