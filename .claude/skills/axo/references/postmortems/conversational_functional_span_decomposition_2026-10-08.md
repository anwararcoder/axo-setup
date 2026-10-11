# Postmortem - Conversational Functional Span Decomposition (2026-10-08, lost)

## A. Challenge facts

- Task: spans [start, end, function] over the tokens of an utterance; micro F1 over exact events; From Scratch ("All learned model parameters must start from random initialization"); 4,248 training utterances in 9 conversations, 1,083 test utterances in 3 conversations; baseline 0.3536.
- Result: best counted score 0.4607 (the same model scored 0.4726 before its determinism fix); 14th when the grace hour ended; tenth place 0.5030, top 0.5388.
- Uploads: v1 sklearn token tagger (local 0.460) refused, compliance; v2 PyTorch sparse models 0.4507 (local 0.481); v3 + recurrent reader 0.4726 scored, held for determinism; v4 same with fixes 0.4607 (local 0.514); v5 three readers (local 0.519) held for determinism, cause not found.
- Method: boundary decision per token gap and function label per span, each from a sparse softmax layer and from a bidirectional GRU reader with word and character inputs and the other speaker's last utterance; logits averaged 0.5 / 0.5.
- Data facts: spans cover 99.9% of tokens; 87% of inner span starts follow punctuation; function right on 62% of exactly matched spans; label mix differs by conversation; "ja." has six functions.
- Validation: leave-one-conversation-out; per-conversation scores 0.36 to 0.62.

## B. Reusable lessons (promoted)

1. The deterministic skeleton and the final framework from the first file (determinism.md, section 8; gate checks added).
2. With three test groups, a local gain under the between-group spread over root three is not evidence.
3. Cache model outputs per fold before scoring many decoding settings.
