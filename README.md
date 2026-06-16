# RNN SMS Spam Detection (Discovery-to-Action)

An LSTM-based Recurrent Neural Network for SMS spam classification using TensorFlow/Keras, structured around the **Discovery → Technical → Action (DTA)** framework.

## How to run

1. Open `rnn_sms_spam.ipynb` in Google Colab (File → Upload notebook, or drag into colab.research.google.com).
2. Runtime → Run all. No GPU needed.
3. The notebook tries to download the real **UCI SMS Spam Collection** dataset (5,574 labeled messages) automatically. If that download fails in your runtime (no internet, host unreachable), it falls back to a synthetic dataset built from realistic spam/ham phrasing patterns, so the notebook still completes end-to-end with no manual steps.
4. To use your own CSV instead, use the "PATH C" cell in Section 1.1.
5. Total runtime: ~2-5 minutes.

## Dataset

- **Primary source:** UCI SMS Spam Collection (5,574 real SMS messages, labeled spam/ham), downloaded directly inside the notebook.
- **Fallback:** synthetic dataset (1,000 messages: 400 spam-template, 600 ham-template) used automatically if the download is unavailable, preserving a realistic class imbalance (spam as the minority class).
- **Label mapping:** `"spam"` → `1`, `"ham"` → `0`.
- **Missing values:** rows with null/empty text dropped (no usable signal to learn from).
- **Split:** 80% train / 20% validation, stratified by label.

## Preprocessing pipeline

1. **Tokenizer** (`keras.preprocessing.text.Tokenizer`, `num_words=10000`, `oov_token="<OOV>"`) — fit only on training text to avoid vocabulary leakage from validation data; unseen words at inference map to a reserved OOV token.
2. **`pad_sequences`** (`maxlen=50`, `padding="post"`, `truncating="post"`) — standardizes every message to a fixed 50-token length so the RNN receives uniform-shaped input; padding/truncating at the end preserves the message opening at a fixed position.

## Model architecture

```
Embedding(input_dim=10,000, output_dim=16, input_length=50)
  -> LSTM(32)
  -> Dropout(0.2)
  -> Dense(16, activation='relu')
  -> Dense(1, activation='sigmoid')
```

- **LSTM over a pooling/averaging approach** — spam cues often depend on word order and context within a sequence, and LSTM's gating mechanism also mitigates vanishing gradients across longer messages, which a simple bag-of-embeddings model cannot capture.
- **Dropout(0.2)** — randomly zeroes 20% of the LSTM's output activations during training to discourage over-reliance on a small set of memorized spam keywords ("free", "winner", "$"), encouraging the model to use broader combinations of signals.
- **EarlyStopping** monitors `val_loss` (not `val_accuracy`) — on an imbalanced dataset, accuracy can look stable even as the model overfits and quietly shifts its false-positive/false-negative balance, while `val_loss` is a more sensitive generalization signal. `patience=3`, `restore_best_weights=True`.
- Loss: `binary_crossentropy` | Optimizer: `adam` | Metrics tracked: `accuracy`, `precision`, `recall`.

## Required triage test

| Message | Expected | Notebook result |
|---|---|---|
| "Hey, are we still meeting for lunch?" | Ham | *(fill in P(spam) from notebook Section 3.1 output)* |
| "URGENT! Your account is locked. Click here to verify." | Spam | *(fill in)* |
| "Congratulations, you won a $500 gift card!" | Spam | *(fill in)* |

## Why precision over accuracy

Accuracy is misleading on an imbalanced dataset like SMS spam — a model that always predicts "ham" can still score a high accuracy without catching any spam at all, simply because spam is the minority class. What matters operationally is the **asymmetric cost of errors**:

- **False positive** (real message flagged as spam) — a missed appointment, 2FA code, or message from a real contact buried in junk. High, often irreversible cost.
- **False negative** (spam let through) — the user sees one extra unwanted text and deletes it. Low, recoverable cost.

**Precision** (`TP / (TP + FP)`) directly measures how trustworthy a "spam" verdict is — exactly the guarantee needed before an automated system silently removes a message from the user's inbox. This is why the model tracks precision and recall explicitly rather than relying on accuracy alone.

## Threshold recommendation

| Threshold band | Action |
|---|---|
| `P(spam) >= 0.95` | Auto-route to Junk folder |
| `0.50 <= P(spam) < 0.95` | Route to a visible "Possible Spam" review folder |
| `P(spam) < 0.50` | Leave in main inbox |

**Why 0.95, not 0.5:** the default 0.5 boundary optimizes for overall correctness, not the cost asymmetry above. Raising the auto-junk bar to 0.95 sacrifices some recall (a few spam messages won't get auto-filed) in exchange for much higher precision on the automated action — the right tradeoff given that a missed spam message costs a few seconds of annoyance, while a real message disappearing into an unchecked junk folder can cost something the user can't recover (a missed call-back, a missed delivery window). The middle "Possible Spam" band keeps the recall the strict threshold gives up from being silently discarded — it becomes a one-tap human decision instead of an automated one.

*(Run the notebook's Section 3.3 precision-recall sweep and replace the table above with your model's actual threshold-vs-precision numbers before submission.)*

## Practical and ethical considerations

- **False positives erode trust fast** — one missed appointment or 2FA code due to over-filtering can cause a user to disable spam filtering entirely, losing all its protection.
- **Sender reputation should complement content-only modeling** — identical wording can be legitimate from a known contact and malicious from a stranger; production systems should combine this text model with sender-level signals.
- **Privacy** — on-device inference is strongly preferable to sending personal SMS content to a server; any real-user training data requires informed consent and anonymization.
- **Vocabulary/tactic drift** — spammers adapt wording to evade filters (e.g. "fr33"); a static model degrades over time without periodic retraining.
- **Legitimate urgency can resemble spam lexically** — "URGENT call me, mom is in the hospital" shares surface features with phishing messages; this is part of why the threshold leans conservative.
- **Equity** — users who depend on SMS for time-sensitive communication (delivery updates, app-less 2FA) bear a disproportionate cost from false positives.
- **Model scope** — this LSTM only sees message text, not sender ID, send frequency, or actual link destinations; a production system would combine it with those signals rather than deploying it standalone.

## Repository contents

- `rnn_sms_spam.ipynb` — full notebook (Discovery → Technical → Action), runs end-to-end in Colab
- `README.md` — this file
