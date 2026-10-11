# Solution description — StanzaOrder: From-Scratch Restoration of Shuffled Verse Lines in Historical German Poetry for Unseen Poets

## Approach

The solution predicts, for every stanza, which line is printed first and which line follows each line. It then picks the order that is most probable as a whole.

1. **Line features**, computed from each line's own text:
   - the punctuation the line ends with (full stop, colon, comma, virgule and so on), since the period prints close a sense unit with a full stop or colon;
   - its first word, as a one-hot over the 100 most frequent line-initial words of `train.csv`, counted in code;
   - its syllable count, and that count relative to the stanza's median;
   - whether it has a masculine or feminine ending;
   - how many other lines of the same stanza it rhymes with;
   - simple counts: commas, question and exclamation marks, virgules, line length.

   Rhyme is detected with hand-written string rules on word endings, which the rules allow; no dictionary is used. The rules normalise the spelling variants of the prints: long s, a/o/u with a small e above, ey/ei/ai, eu/äu, th, dt, tz, doubled letters and final devoicing. Two lines rhyme when their last words agree from the last stressed vowel on, which is one vowel earlier for a feminine ending.
2. **Pair features** for every ordered pair of lines of the stanza: they rhyme, they share the last two or three sound letters, the last words are identical, they have the same ending type, and their syllable difference.
3. **A small neural network**, randomly initialised and trained in the script on `train.csv` only, maps these features to three distributions per stanza:
   - which line comes first;
   - for every line, which line follows it;
   - for every line, whether the stanza ends after it.

   It has a shared two-layer line encoder, a pair scorer and two linear heads, and is trained with cross-entropy against the printed orders: 30 epochs, AdamW, one-cycle schedule.
4. **Decoding.** The order with the largest total log-probability (first line, plus every successor, plus the end) is found exactly by dynamic programming over subsets of lines (Held-Karp). That is at most 2^10 × 10 states for a 10-line stanza, so every stanza gets its exact optimum.

Each test stanza is featurised and decoded on its own. The only shared state is what was fitted on `train.csv`: the first-word vocabulary and the network weights.

## Validation

- **Method.** Poets are held out whole, as in the test split. The training poets are split into five folds of whole poets, and fold 0 (22 poets, 2,075 stanzas) is held out. The vocabulary and the network are fitted on the other 90 poets only.
- **Result on the held-out poets:** mean row score 0.4205. Reweighted to the test set's mix of stanza sizes (4 to 10 lines) it is **0.3886**.
- **For comparison:** a random order scores 0.1845 on the test mix, the line-label order 0.2124 on this hold-out, and the fit on the training poets is 0.5066.
- **Noise:** the per-row standard deviation is 0.371, so one standard error on this hold-out is about 0.008.

This is a local estimate only.

## Rule compliance

- **From scratch:** no pretrained weights or embeddings, no external text, no poetry editions, no rhyme or pronunciation dictionaries. Every learned parameter is created by random initialisation in the script and trained there on the provided `train.csv`. The rhyme rules are hand-written string rules.
- **Test rows:** `test.csv` is read only after training, for the single prediction pass. Each test stanza is predicted from its own lines alone. Stanzas of the same size are passed through the dynamic programme together for speed, but each one is solved independently.
- **Determinism:** one fixed device (`cuda:0`), fixed seeds, `torch.use_deterministic_algorithms(True)`, cuDNN deterministic with benchmarking off, TF32 off, a fixed cuBLAS workspace, a fixed number of epochs, no early stopping, no data-loader workers, no mixed precision. Vocabularies are sorted, and nothing depends on set iteration order or Python's hash seed. Two full runs gave byte-identical submissions.
- **Output:** one row per test `example_id` in the sample's order, each a permutation of that stanza's labels. The script writes a valid fallback file (the sample) at start, and replaces it at the end.

## Runtime

0.3 minutes for the full script (featurisation, training and decoding), measured in two separate runs on an NVIDIA RTX 3050 Laptop GPU. That is far inside the one-hour runtime target.
