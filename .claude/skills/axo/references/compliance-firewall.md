# Compliance firewall and audit pass

Source: the challenge-engine pack (`references/pack/`), the trusted source where it and older text disagree. The challenge description is the source of truth; this page is how it is enforced.

## Evaluation-integrity firewall

Run it at intake, before every design change, and before every submission. Reject a technique unless the description clearly allows it:

- training any model or language model on the test set's context, forms or examples; test rows as training inputs;
- test-to-test information flow: pooling context from other evaluation cases, using sibling or equivalent cases to infer a held-out target, retrieval among test cases, statistics computed across test rows;
- labels or pseudo-labels derived from the evaluation population; self-training; test-time augmentation or any other use of the test set beyond the single prediction pass, when the rules forbid it;
- synthetic labelled examples (transformations, paraphrases, "disputes") when synthetic data is prohibited;
- hidden or test labels, external labels, post-hoc annotations;
- fitting, normalising or choosing preprocessing, vocabulary, thresholds, calibration, embeddings, retrieval indexes, encoders or hyper-parameters on evaluation data; letting validation or test identity into feature engineering, retrieval or model selection;
- the public/private split, row order, ids or released group aliases as a signal when the rules say not to depend on them (a validation-group alias released in `test.csv` is for grouped validation only; do not use it for "harmless" normalisation either);
- an external resource that effectively is an annotation source.

## Six questions

Ask them at intake and again before each submission; write the answers into `RULES.md`.

1. Which exact rows may be used for training?
2. What information may be used at inference time?
3. Can any feature depend on another evaluation example?
4. Can the model or preprocessing see the target population before prediction?
5. Is any generated data forbidden?
6. Is any external resource effectively an annotation source?

When uncertain, take the conservative interpretation and tell the user about the uncertainty. Never take the risky path silently; never invent permissions, data access, model allowances or evaluation behaviour.

## Audit pass

An independent read of the rules against the code, in this order of priority:

1. rule compliance; 2. evaluation leakage; 3. split correctness; 4. train/test information flow; 5. runtime and determinism hazards (`determinism.md`, bad patterns); 6. obvious generalisation risks; 7. submission format.

- **When:** before the first submission and before the finalized one. Not before a user's "now" hand-over, and not while the user waits at 8/10: hand over first, audit while they upload if time allows.
- **How:** a second, deliberate pass over the exact files being sent; a fresh-context subagent when the user asks for one. Return only actionable findings with evidence from the rules or the code. Do not redesign the solution unless a finding requires it.
- **Output:** findings ranked by the order above, each with the rule or line it rests on; "no findings" is a valid result and says what was checked.

## The challenge markdown decides, and people check it after scoring (The Enclosure Mention, rejected)

The user's account, 2026-10-07: the best version of The Enclosure Mention (0.835) was scored by the bot and later rejected by human reviewers for one inference step that let one test row change another's answer when their cover texts matched. The reviewers' text: "[R]ejected. Final inference modifies a pair's candidate probabilities using another test pair's probabilities when their covers match. Each prediction must use its own cover–enclosure pair and training-fitted state. Remove the cross-pair adjustmen[t]"

The challenge markdown had said not to do this, in its own wording. The step stayed in because a version containing it had been scored, which was read as acceptance. It was not: there is a review phase after scoring that nobody had known about.

- The source of the rules is the markdown of the challenge in hand. Each challenge has its own rules; what one allows another forbids. Do not import a permission or a prohibition from another challenge or from habit: look it up in this markdown.
- The firewall list above is a set of questions to put to that markdown, not a fixed law. For each item, find the sentence in the markdown that settles it and quote it in `RULES.md`. Where the markdown is strict, be strict to the letter.
- A score from the bot proves the file was valid, nothing more. The code is read by people afterwards against the markdown. Check every hand-over as if that review were happening now.
- If a sentence in the markdown can be read as ruling a step out, the step is out.
