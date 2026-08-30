# Phase 4 Deep Dive — The Capstone: a Character-Level GPT

Companion notes for **Phase 4** of `pytorch-char-gpt-guide.md`.
Detailed explanations + runnable examples for the phase where everything converges: you build a transformer language model from scratch, implement self-attention by hand, and end up with a tiny relative of every modern LLM.

This picks up where the Phase 3 deep dive left off. You own the full training method: baseline → diagnose → one change → measure → write it down. You own AdamW, cosine scheduling, dropout, AMP, and — critically — residual connections. **All of it returns here.** What's new is the *problem shape*: sequences instead of images, "predict the next character" instead of "name the class," and an architecture whose one genuinely new idea is self-attention. Everything else in the GPT is a part you've already built.

> **How to read this.** Phase 4 has its own rhythm again: **six incremental steps, each one runnable.** The discipline of the phase is *never write the whole thing at once*. Each step produces a working program whose validation loss you record; by the end you'll have a ledger showing exactly how much each architectural idea was worth. If a step doesn't run, you know the bug is in the ~30 lines you just added — that's the entire debugging strategy, and it's why people who build it incrementally can debug transformers while people who paste nanoGPT cannot. Keep Karpathy's *"Let's build GPT"* video alongside: watch a section, build it yourself, then diff your understanding against his code.

> **How the code in this file works — read this once.** Every step presents its code twice, on purpose:
>
> 1. **Teaching snippets** inside the prose. These show *only the new idea* and are deliberately not standalone — they assume everything from earlier steps is already in your file, and some are openly incomplete (`# ... same loss code as Step 2 ...`). Type them to feel the idea; don't try to run them alone.
> 2. **"The file so far"** at the end of each step: the **complete, copy-paste-runnable `gpt.py`** at that moment, with `# NEW` and `# CHANGED` comments marking exactly what this step added. This is the ground truth. If a snippet and the full file ever seem to disagree, the full file wins.
>
> So the loop for each step is: read the prose → write the new part yourself → diff your file against "the file so far" → run it → do the break-it experiments. And **save a copy of each step before moving on** (`git commit -am "step 3"`, or `cp gpt.py steps/s3_onehead.py`). You'll want to re-run earlier steps to compare losses, and each saved file is a known-good rollback point for when Step 5 mysteriously stops training.

---

## The mental model for the whole phase

Strip away the mystique first. A language model is a **classifier**. Phase 2's classifier looked at an image and produced 10 logits, one per class. This model looks at a sequence of characters and produces `vocab_size` logits — one per character it might see *next*. Same `CrossEntropyLoss`, same training loop, same optimizer. If Tiny Shakespeare has 65 unique characters, you are building a 65-class classifier.

Two things make it feel different, and both are just bookkeeping:

1. **The model predicts at every position at once.** Feed in a chunk of `T` characters and it emits `T` next-character predictions — position 3's prediction uses characters 1–3, position 7's uses characters 1–7. One training chunk = `T` training examples for free. This is why the logits tensor is `(B, T, vocab_size)` instead of Phase 2's `(B, 10)`, and it's the source of most Phase 4 shape errors, so pin the convention now:

   ```
   B = batch size        (how many independent chunks, e.g. 32)
   T = time / sequence   (how many characters per chunk, up to block_size)
   C = channels          (embedding dimension, or vocab_size at the output)
   ```

   Every tensor in this phase is some slice of `(B, T, C)`. When you're lost, ask "which of B, T, C is this dimension?" — it's the Phase 1 shape-printing habit with a schema attached.

2. **Generation is the model eating its own output.** Predict a distribution over the next character, *sample* from it, append the sample to the input, repeat. That loop — the same one behind every chat model you've used — is about ten lines, and you'll write it in Step 2, before attention even exists.

And the arc of the six steps, as a loss ledger you'll fill in with your own numbers:

| Step | What exists | Val loss (≈, Tiny Shakespeare) | What the drop proves |
|---|---|---|---|
| 2 | Bigram (embedding lookup only) | ~2.5 | The pipeline works; one char of context is worth this much |
| 3 | + one attention head | ~2.4 | Talking to the past helps |
| 4 | + multi-head + feed-forward | ~2.2 | Parallel perspectives + per-token computation help |
| 5 | + residual blocks, LayerNorm, depth | ~2.0 | Depth is trainable now (Phase 3.8's lesson, cashed in) |
| 6 | scaled up (`n_embd=384`, 6 layers, dropout) | **~1.5** | Scale works when the architecture is right |

For calibration: a model that knows *nothing* assigns uniform probability `1/65` to every character, giving loss `−ln(1/65) ≈ 4.17`. That is your reference point forever: everything below 4.17 is knowledge. (Step 2 adds one small caveat about reading it at initialization.)

One workflow note before code: keep everything in **one file, `gpt.py`**, growing step by step (Phase 5 splits it into modules — that refactor is a lesson of its own, don't preempt it). Name TensorBoard runs by step: `runs/s2_bigram`, `runs/s3_onehead`, ... — the ledger above should fall out of your TensorBoard overlay by the end.

---

# 4.1 Step 1 — Data and the language-modeling setup

## Get the corpus

Tiny Shakespeare is ~1.1 MB of plain text — every Shakespeare play concatenated. Download it once into `data/` (already git-ignored):

```python
import os, urllib.request

os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(url, "data/input.txt")

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(len(text))          # ~1,115,394 characters
print(text[:200])         # First Citizen: / Before we proceed any further...
```

Why this corpus is the standard: it's small enough to train on in minutes, large enough that a small model can't just memorize it, and Shakespeare's style is so distinctive that you can *see* the model learning it — the generated text tells you how training went in a way a loss number can't.

## The tokenizer — 65 characters, two dictionaries

A tokenizer converts text to integers and back. Real LLMs use subword tokenizers (BPE) with vocabularies of ~50k–200k tokens; we use the simplest possible scheme — **one token per character** — so the tokenizer never obscures the model:

```python
chars = sorted(set(text))
vocab_size = len(chars)
print("".join(chars))     # \n !$&',-.3:;?ABCDEFGH... abcdefgh...
print(vocab_size)         # 65

stoi = {ch: i for i, ch in enumerate(chars)}   # string -> int
itos = {i: ch for ch, i in stoi.items()}       # int -> string

def encode(s):  return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

print(encode("hii there"))            # [46, 47, 47, 1, 58, 46, 43, 56, 43]
print(decode(encode("hii there")))    # hii there
```

`decode(encode(s)) == s` for any `s` made of corpus characters — that round-trip is the contract, and it becomes your first unit test in Phase 5. Note the trade-off you just made: a tiny vocabulary (65 vs 50,000) in exchange for long sequences (one token per *character*, so "the" costs 3 tokens instead of 1). Char-level is the right end of that trade for learning; Phase 5's BPE swap explores the other end.

## Encode everything, split train/val

```python
import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)     # torch.Size([1115394]) torch.int64

n = int(0.9 * len(data))
train_data = data[:n]             # first 90%
val_data   = data[n:]             # last 10%
```

Two details that matter:

- **`dtype=torch.long`** (int64) because these integers will index into an `nn.Embedding`, and PyTorch requires long indices. Feeding floats to an embedding is a classic first error of the phase.
- **The split is a contiguous cut, not a shuffle.** Shuffling characters would destroy the sequences we're trying to model. Cutting the tail off means the val set is *text the model has never read*, which is exactly what "validation" should mean here. (The model can still overfit style; it can't memorize the exact lines.)

## `block_size`, and what one training example actually is

`block_size` is the maximum context length — how far back the model can ever look. Take a chunk of `block_size + 1` characters:

```python
block_size = 8                       # tiny for now; 256 in Step 6
x = train_data[:block_size]          # inputs
y = train_data[1:block_size + 1]     # targets = inputs shifted one to the right
for t in range(block_size):
    print(f"context {x[:t+1].tolist()} -> target {y[t].item()}")
```

Output (with Tiny Shakespeare's opening "First Ci..."):

```
context [18] -> target 47
context [18, 47] -> target 56
context [18, 47, 56] -> target 57
context [18, 47, 56, 57] -> target 58
...
```

**Stare at this until the shift makes sense** — the guide means it. One chunk of 9 characters contains **8 supervised examples**, from "given 1 character, predict the 2nd" up to "given 8, predict the 9th." The model trains on all context lengths simultaneously, which is also why it can *generate* from a context of any length up to `block_size`. `y` is just `x` shifted left by one; the "labels" are the text itself. This is **self-supervision** — no human labeled anything, which is precisely why this recipe scales to the entire internet.

## `get_batch` — the whole data pipeline in six lines

No `Dataset`, no `DataLoader`. The corpus is one tensor in memory, so a "batch" is just `B` random chunks:

```python
torch.manual_seed(1337)
batch_size = 4          # B — chunks per batch (32 in Step 6)
block_size = 8          # T — characters per chunk (256 in Step 6)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))       # B random offsets
    x = torch.stack([data[i     : i+block_size    ] for i in ix])   # (B, T)
    y = torch.stack([data[i + 1 : i+block_size + 1] for i in ix])   # (B, T)
    return x.to(device), y.to(device)

xb, yb = get_batch("train")
print(xb.shape, yb.shape)    # torch.Size([4, 8]) twice
print(xb)
print(yb)                    # xb shifted one to the left, with one new char at the end
```

Print a batch and check the shift by eye: row 0 of `yb` should be row 0 of `xb` minus its first element, plus one new element at the end.

Notice the shift in training vocabulary this causes: there are no epochs anymore. `get_batch` samples random positions with replacement, so we count progress in **iterations** (steps), not passes over the data. `max_iters` replaces `num_epochs`, and "one more epoch" becomes "another few thousand steps." Nothing deep — just bookkeeping to expect in the training loop.

## The file so far — `gpt.py` after Step 1

Everything above, assembled. No model yet: this file's whole job is to turn a text file into `(x, y)` tensors on the right device, and to prove it did so correctly. Run it with `python gpt.py`.

```python
"""
gpt.py — Step 1: data and the language-modeling setup.
Run: python gpt.py   ->  corpus stats, one batch, and the examples hiding inside it.
"""
import os
import urllib.request
import torch

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config ----------------
batch_size = 4        # B — chunks per batch      (32 from Step 2 on)
block_size = 8        # T — characters per chunk  (256 in Step 6)

# ---------------- data ----------------
os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(url, "data/input.txt")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ---------------- tokenizer ----------------
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}      # string -> int
itos = {i: ch for ch, i in stoi.items()}          # int -> string
def encode(s):  return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

# ---------------- train/val split ----------------
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i     : i+block_size    ] for i in ix])    # (B, T)
    y = torch.stack([d[i + 1 : i+block_size + 1] for i in ix])    # (B, T)
    return x.to(device), y.to(device)

# ---------------- smoke test ----------------
if __name__ == "__main__":
    print("device:", device)
    print(f"{len(text):,} characters | vocab_size={vocab_size}")
    print("".join(chars))
    assert decode(encode(text[:1000])) == text[:1000], "tokenizer round-trip broken"

    xb, yb = get_batch("train")
    print(xb.shape, yb.shape)                    # torch.Size([4, 8]) twice
    print(repr(decode(xb[0].tolist())))          # a real Shakespeare fragment
    print(repr(decode(yb[0].tolist())))          # ...the same, shifted one left
    for t in range(block_size):                  # the 8 examples hiding in row 0
        print(f"context {xb[0, :t+1].tolist()} -> target {yb[0, t].item()}")
```

Everything in this file survives to Step 6 essentially untouched — the config values change and a model gets bolted on top, but `get_batch` is the last version of `get_batch` you will write.

### Break-it experiments — 4.1

- **Decode a batch row:** `print(decode(xb[0].tolist()))` — see actual Shakespeare mid-sentence. Do it for `yb[0]` too and *see* the one-character shift in text form.
- **Round-trip test:** `assert decode(encode(text[:1000])) == text[:1000]`. Then try `encode("hello, world!")` — it works — and `encode("héllo")` — `KeyError`, because `é` isn't in the corpus. Your tokenizer's vocabulary is a hard boundary; real tokenizers solve this with byte-level fallbacks.
- **Count the examples:** one batch of `(B=4, T=8)` contains 32 supervised prediction problems. The Step 6 config `(32, 256)` contains 8,192 per batch. Cheap supervision at scale is the whole game.
- **Peek at the class balance:** `torch.bincount(data, minlength=vocab_size)` — spaces and `e` dominate, capital `Z` barely exists. Remember Phase 2's balanced 10 classes? This is your first *imbalanced* problem, and it's partly why generated text gets spaces and vowels right first.

---

# 4.2 Step 2 — The bigram baseline (no attention yet)

## The dumbest possible language model

A bigram model predicts the next character from **the current character only** — no context, no memory. And there's a beautifully direct way to build it: an embedding table of shape `(vocab_size, vocab_size)`, where **row `i` holds the logits for what follows character `i`**. Looking up a character *is* the entire forward pass:

```python
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):            # idx: (B, T) of ints
        logits = self.token_embedding(idx)           # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
```

Two things here are load-bearing for the rest of the phase:

**`nn.Embedding` is a lookup table, nothing more.** `nn.Embedding(65, 65)` is a `(65, 65)` matrix of learnable numbers; indexing with an int tensor of shape `(B, T)` returns the corresponding rows, shape `(B, T, 65)`. No matmul, no activation. In Step 5 the same layer reappears as `nn.Embedding(vocab_size, n_embd)` — same mechanics, but the rows become learned *representations* of each character rather than direct logits.

**The cross-entropy reshape.** `F.cross_entropy` wants `(N, C)` predictions and `(N,)` targets — it doesn't know about time. But we established that a `(B, T)` batch is really `B*T` independent classification problems, so we flatten exactly that way: `logits.view(B*T, C)`, `targets.view(B*T)`. This one reshape is the entire adaptation of Phase 2's loss to sequences. (Why not `logits.view(-1)`? Print the shapes if the view arithmetic feels shaky — Phase 1's golden rule never retires.)

## `generate` — the loop you'll never rewrite

This is a **method on `BigramLM`** — note the indentation; it goes inside the class, below `forward`. (See the full file at the end of this step for it in place.)

```python
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):         # idx: (B, T) — the prompt
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                    # (B, T, C)
            logits = logits[:, -1, :]                # last position only: (B, C)
            probs = F.softmax(logits, dim=-1)        # logits -> probabilities
            idx_next = torch.multinomial(probs, 1)   # SAMPLE one char: (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # append and go again
        return idx
```

Walk it: forward the whole context, keep only the **last** position's logits (the prediction for "what comes next"), softmax into a distribution, **sample** with `torch.multinomial`, append, repeat. Sampling — not argmax — is a real decision: taking the argmax every time produces repetitive, loop-prone text ("the the the"); sampling respects the model's uncertainty and gives varied output. Step 6's temperature knob tunes exactly this trade-off.

(Yes, for a bigram model, re-forwarding the whole growing sequence just to use the last position is wasteful. Keep it anyway — this exact method works **unchanged** for the full transformer in Step 5, which genuinely needs the whole context. Writing the general version now is why the guide can say "reuse your Step-2 `generate()` unchanged.")

## Train it and calibrate your expectations

The loop below assumes `device`, `get_batch`, and `vocab_size` from Step 1 are already in the file — it is a fragment, not a program:

```python
model = BigramLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10000):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 1000 == 0:
        print(step, loss.item())
```

Before you run it, **predict the first loss**. A model that knows nothing spreads probability uniformly over 65 characters: `−ln(1/65) ≈ 4.17`. That's the reference number for the whole phase — but read it with one correction at this step. `nn.Embedding` initializes its table from a standard normal, and in the bigram model those numbers *are* the logits, so the untrained distribution isn't uniform, it's randomly lumpy — which costs you a few tenths. **Expect ~4.6–4.9 here** (Karpathy's run prints 4.87). From Step 3 on, logits come out of a `nn.Linear` whose default init is much smaller, the initial distribution really is near-uniform, and you'll see **~4.17 on the nose**.

Either way it's the LM version of Phase 3's "overfit a single batch" sanity check, and the first number to read in *every* step from here on. Way above ~5 at init → bug, often in the reshape. Exactly 0 → you're leaking targets somewhere.

The loss will fall to about **2.5 and stop**. It must: a bigram model looking at `q` can learn "probably `u` next," but looking at ` ` (space) it faces the full entropy of "which word starts here?" — and no amount of training fixes that, because the *information isn't in its input*. That floor is an architecture limit, not an optimization failure. Distinguishing those two is the core diagnostic skill of this phase (it's 3.3's "diagnose before treating," transposed).

Generate from it — the convention is to prompt with token 0, which for this corpus is `\n`:

```python
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 300)[0].tolist()))
```

You'll get something like `"Foanghand ce llo t athe soff hen&whar..."` — near-gibberish with *suspiciously plausible texture*: reasonable word lengths, vowels in the right density, almost-words. That's a bigram model working perfectly. **Do not skip savoring this moment:** the full pipeline — data → model → loss → generation — is done. Every remaining step only changes what happens between the embedding and the logits.

## The file so far — `gpt.py` after Step 2

Step 1's file with three things added: the `BigramLM` class (including `generate` as a **method**, which is what the indentation in the snippets above was signalling), a training loop, and `estimate_loss`.

That last one is a small jump ahead: §4.6 explains in full why you average many batches over both splits instead of trusting a single noisy batch. You need it from here on, because the ledger you're filling in wants a **val** loss at every step — so it arrives now, and you can read its justification later.

```python
"""
gpt.py — Step 2: the bigram baseline (no attention yet).
New vs Step 1: BigramLM, the training loop, estimate_loss, TensorBoard, generation.
Run: python gpt.py   ->  loss 4.17 -> ~2.5, then 300 characters of textured gibberish.
"""
import os
import math                                        # NEW
import urllib.request
import torch
import torch.nn as nn                              # NEW
import torch.nn.functional as F                    # NEW
from torch.utils.tensorboard import SummaryWriter  # NEW

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config ----------------
batch_size    = 32          # CHANGED (was 4): we're really training now
block_size    = 8
learning_rate = 1e-3        # NEW
max_iters     = 10000       # NEW — iterations, not epochs (see 4.1)
eval_interval = 500         # NEW
eval_iters    = 50          # NEW — batches averaged per loss estimate
run_name      = "runs/s2_bigram"   # NEW — one run dir per step; the ledger falls out of this

# ---------------- data ----------------  (identical to Step 1)
os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(url, "data/input.txt")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s):  return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i     : i+block_size    ] for i in ix])
    y = torch.stack([d[i + 1 : i+block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- model ----------------  (all NEW)
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # row i = the logits for whatever follows character i
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):            # idx: (B, T) of ints
        logits = self.token_embedding(idx)           # (B, T, vocab_size)
        loss = None
        if targets is not None:                      # note: `loss = None` + a single
            B, T, C = logits.shape                   # return OUTSIDE the if — generate()
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss                          # calls forward with no targets

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):         # idx: (B, T) — the prompt
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                    # (B, T, C)
            logits = logits[:, -1, :]                # last position only: (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)   # SAMPLE one char: (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # append and go again
        return idx

# ---------------- eval ----------------  (NEW — explained in §4.6)
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------- train ----------------  (NEW)
if __name__ == "__main__":
    model = BigramLM(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(run_name)
    # uniform-guess loss. This model starts ABOVE it (~4.6-4.9) because nn.Embedding
    # inits from N(0,1) and those numbers are the logits; Step 3 onward lands on it.
    print(f"uniform-guess loss: {-math.log(1/vocab_size):.2f}")

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            writer.add_scalar("loss/train", losses["train"], it)
            writer.add_scalar("loss/val",   losses["val"],   it)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.close()

    # ---------------- generate ----------------
    start = torch.zeros((1, 1), dtype=torch.long, device=device)   # token 0 == "\n"
    print(decode(model.generate(start, 300)[0].tolist()))
```

Two gotchas this file is quietly protecting you from, both of which bite people who assemble the snippets by hand:

- **`return logits, loss` sits outside the `if targets is not None` block.** Indent it one level too far and `forward` returns `None` whenever targets are absent — which is exactly what `generate` does — so you get `TypeError: cannot unpack non-sequence NoneType` on the first generated character.
- **`torch.multinomial(probs, 1)` takes the sample count.** `torch.multinomial(probs)` is a `TypeError`, and `torch.mutlinomial` is the typo everyone makes at 1am.

### Break-it experiments — 4.2

- **Generate before training** (fresh model). Uniform soup: `"$Zk?3jNx..."`. Compare with post-training output — that difference is what loss 4.17 → 2.5 *looks like*.
- **Inspect the learned table.** `F.softmax(model.token_embedding.weight[stoi['q']], dim=-1)` and print the top-5 with `itos`. `u` should dominate. Do the same for `' '` and see the entropy problem directly: probability smeared over every letter that can start a word.
- **Prove the floor is real:** train 5× longer, add a scheduler, triple the LR — val loss will not meaningfully beat ~2.45. Underfitting that more optimization can't fix = the model class is the bottleneck. (Now regularization intuition from 3.3: would dropout help here? No — this model isn't overfitting, it's *incapable*.)
- **Overfit check in miniature:** train on a single fixed batch for 500 steps — loss should crash toward ~0 (it *can* memorize 32 chunks). Pipeline-can-learn-at-all, confirmed, same as Phase 3.2.

---

# 4.3 Step 3 — A single self-attention head

This is the centerpiece of the capstone, and the one genuinely new idea in the whole phase. Budget real time here. Everything is done on tiny tensors (`B=1, T=8`) that you print and stare at — the GPU is irrelevant in this section.

## The problem, stated plainly

The bigram model failed because position `t` predicted from **only** token `t`. What we want: position `t` should gather information from **all previous positions** — but *not future ones*, since those contain the answer it's trying to predict. So the question is: how does a token look back and combine information from its past?

## Warm-up: the cheapest possible "look back" — averaging

Forget queries and keys for a moment. The crudest aggregation is a plain average: let each position's vector be the **mean of all vectors up to and including it**. No learning, just "blend the past into the present." Here's the loop version, then the insight that makes transformers fast:

```python
torch.manual_seed(1337)
B, T, C = 1, 8, 2                    # tiny on purpose
x = torch.randn(B, T, C)

# version 1: explicit loops (slow, but unambiguous)
xbow = torch.zeros(B, T, C)          # "bag of words" average
for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, :t+1].mean(dim=0)
```

Now the trick: **that double loop is a matrix multiply by a lower-triangular weight matrix.**

```python
wei = torch.tril(torch.ones(T, T))          # lower-triangular ones
wei = wei / wei.sum(dim=1, keepdim=True)    # normalize each row to sum to 1
print(wei)
# [[1.0000, 0.0000, 0.0000, ...],
#  [0.5000, 0.5000, 0.0000, ...],
#  [0.3333, 0.3333, 0.3333, ...],
#  ...
xbow2 = wei @ x                             # (T,T) @ (B,T,C) -> (B,T,C), batched matmul
print(torch.allclose(xbow, xbow2))          # True
```

Read `wei` as a table of **"how much does position t (row) listen to position s (column)?"** Row 2 says: position 2 listens ⅓ each to positions 0, 1, 2, and 0 to everything after — the zeros above the diagonal *are* the "no peeking at the future" rule. Every attention mechanism ever built is this matmul; the entire innovation is in **how `wei` gets computed.**

One more rewrite, because it's the exact form attention uses — get the same matrix via softmax over masked scores:

```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)                              # scores: all equal (for now)
wei = wei.masked_fill(tril == 0, float("-inf"))      # future scores -> -inf
wei = F.softmax(wei, dim=-1)                         # softmax turns -inf into exactly 0
```

Why `-inf` and not just 0? Because `softmax` exponentiates: `exp(-inf) = 0` gives future positions **exactly zero weight** while the surviving scores renormalize among themselves. Masking *before* the softmax is what makes the causality airtight. And crucially: the scores going into the softmax can now be **anything** — the moment we compute them from data instead of leaving them at zero, uniform averaging becomes attention.

## Queries, keys, values — data-dependent listening

Uniform averaging treats every past token as equally interesting. But if the current character is `u`, the recent `q` matters more than a space five tokens ago. Self-attention lets each token *choose* its weights, via three learned projections of the same input (hence **self**-attention):

- **query** `q = x @ W_q` — "what am I looking for?"
- **key** `k = x @ W_k` — "what do I contain?"
- **value** `v = x @ W_v` — "what will I contribute if attended to?"

The score between positions `t` and `s` is the dot product `q_t · k_s` — high when what `t` seeks matches what `s` offers. The full head:

```python
class Head(nn.Module):
    """One head of causal self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, T, n_embd)
        B, T, C = x.shape
        q = self.query(x)                                  # (B, T, head_size)
        k = self.key(x)                                    # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,hs)@(B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                       # rows sum to 1
        wei = self.dropout(wei)
        v = self.value(x)                                  # (B, T, head_size)
        return wei @ v                                     # (B, T, T) @ (B, T, hs) -> (B, T, hs)
```

Line by line, the parts that deserve a pause:

- **`q @ k.transpose(-2, -1)`** computes *all pairwise scores at once*: entry `(t, s)` of the resulting `(B, T, T)` matrix is `q_t · k_s`. `transpose(-2, -1)` swaps the last two dims so the matmul lines up — check the shape comment; this is the single most shape-error-prone line of the phase.
- **`* k.shape[-1]**-0.5` — the "scaled" in scaled dot-product attention.** Dot products of two random `head_size`-dim vectors have variance ≈ `head_size`, so raw scores get large as heads get wide — and softmax of large-magnitude inputs collapses toward one-hot (nearly all weight on one position), which starves every other position of gradient. Dividing by `√head_size` keeps score variance ≈ 1, keeping the softmax in its soft, trainable regime — especially critical at init. One multiply, and training stability depends on it (break-it below).
- **`register_buffer("tril", ...)`** — the mask isn't a parameter (nothing to learn — it's a rule), but it must ride along with the module: `model.to(device)` moves buffers, `state_dict()` saves them, and the optimizer ignores them. A plain attribute tensor would sit on the CPU and crash the first GPU forward. This is the canonical use of a buffer; remember it. The `[:T, :T]` slice lets one `block_size²` mask serve any sequence length up to `block_size` — which matters during generation, when `T` grows from 1.
- **`wei @ v` — the payoff.** The softmaxed `wei` is exactly the warm-up's "listening table," except the model now *computed* it from content. Multiplying by `v` takes, for each position, a weighted blend of what past positions chose to offer. Note that what gets aggregated is `v`, not `x` — a token can advertise one thing (key) and deliver another (value), and that separation is learned.

A phrase worth keeping from Karpathy: **attention is communication** — tokens exchanging information across the sequence. Nothing in the head "thinks"; it *routes*. The thinking is Step 4's feed-forward. Also file away what attention is *not*: it has no notion of position (a fact that becomes a real bug we fix in Step 5), and the communication is one-directional in time only because *we* masked it — the decoder-style mask is a modeling choice, not a law.

## Wire it into the bigram model and measure

Minimal integration for now (the clean version comes in Step 5): give tokens a real embedding, run one head over it, project to logits.

```python
n_embd = 32
dropout = 0.0     # nothing to regularize yet

class OneHeadLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.sa_head = Head(n_embd)                     # head_size = n_embd for now
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)                   # (B, T, n_embd)
        x = self.sa_head(x)                             # (B, T, n_embd) — now context-aware
        logits = self.lm_head(x)                        # (B, T, vocab_size)
        # ... same loss code as Step 2, and generate() copied across unchanged ...
```

(Spelled out in full at the end of this step — including one change `generate` needs now, which the snippets don't show.)

Train with the same loop. Val loss: **~2.5 → ~2.4**. A modest, *real* improvement — one head with a 32-dim embedding is a small brain. The point of this step was never the number; it's that you now hold every moving part of attention in your head, at printable scale.

## The file so far — `gpt.py` after Step 3

Same file, with the model section swapped: `BigramLM` is gone, `Head` and `OneHeadLM` take its place. Everything above the model section (data, tokenizer, `get_batch`) and everything below it (`estimate_loss`, the training loop) is **byte-identical to Step 2** — only the `# ---- model ----` block and two config lines change. That's the whole point of building it this way.

```python
"""
gpt.py — Step 3: one head of causal self-attention.
New vs Step 2: n_embd/dropout config, the Head module, OneHeadLM replacing BigramLM.
Run: python gpt.py   ->  val ~2.5 (bigram) becomes val ~2.4.
"""
import os
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config ----------------
batch_size    = 32
block_size    = 8
n_embd        = 32          # NEW — tokens now get a real representation, not raw logits
dropout       = 0.0         # NEW — nothing to regularize yet; the plumbing goes in now
learning_rate = 1e-3
max_iters     = 5000        # CHANGED (was 10000): this model gets there faster
eval_interval = 500
eval_iters    = 50
run_name      = "runs/s3_onehead"   # CHANGED

# ---------------- data ----------------  (unchanged from Step 1)
os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(url, "data/input.txt")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s):  return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i     : i+block_size    ] for i in ix])
    y = torch.stack([d[i + 1 : i+block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- model ----------------
class Head(nn.Module):                                     # NEW
    """One head of causal self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):                                  # x: (B, T, n_embd)
        B, T, C = x.shape
        q = self.query(x)                                  # (B, T, head_size)
        k = self.key(x)                                    # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5  # (B,T,hs)@(B,hs,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                       # rows sum to 1
        wei = self.dropout(wei)
        v = self.value(x)                                  # (B, T, head_size)
        return wei @ v                                     # (B,T,T)@(B,T,hs) -> (B,T,hs)

class OneHeadLM(nn.Module):                                # NEW (replaces BigramLM)
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)   # -> n_embd, not vocab_size
        self.sa_head = Head(n_embd)                               # head_size = n_embd for now
        self.lm_head = nn.Linear(n_embd, vocab_size)              # NEW: embd -> logits

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)                # (B, T, n_embd)
        x = self.sa_head(x)                          # (B, T, n_embd) — now context-aware
        logits = self.lm_head(x)                     # (B, T, vocab_size)
        loss = None
        if targets is not None:                      # identical to Step 2 from here down
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # NEW vs Step 2, and mandatory now:
            logits, _ = self(idx_cond)               # the head's `tril` is (block_size,
            logits = logits[:, -1, :]                # block_size), so a longer T makes
            probs = F.softmax(logits, dim=-1)        # masked_fill blow up on shapes.
            idx_next = torch.multinomial(probs, 1)   # Step 5 needs this crop for a
            idx = torch.cat([idx, idx_next], dim=1)  # second reason (position embeddings).
        return idx

# ---------------- eval ----------------  (unchanged from Step 2)
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------- train ----------------  (unchanged from Step 2 except the class name)
if __name__ == "__main__":
    model = OneHeadLM().to(device)                       # CHANGED
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(run_name)
    print(f"expected loss at init: {-math.log(1/vocab_size):.2f}")   # ~4.17, and this one hits it

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            writer.add_scalar("loss/train", losses["train"], it)
            writer.add_scalar("loss/val",   losses["val"],   it)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.close()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, 300)[0].tolist()))
```

Note what the embedding table now means. In Step 2, `nn.Embedding(65, 65)` produced logits directly. Here it's `nn.Embedding(65, 32)`: the rows are learned 32-dimensional *representations*, attention mixes them across time, and a separate `lm_head` linear layer turns the result into 65 logits. Same layer, promoted from "answer" to "representation" — and it stays in that role through Step 6.

### Break-it experiments — 4.3

- **Print the attention matrix.** Inside `forward`, after the softmax: `print(wei[0])` for a small batch. Confirm: lower-triangular, every row sums to 1, row 0 is `[1, 0, 0, ...]` (the first token can only attend to itself). This is the "confirm it's lower-triangular" moment the guide insists on — do it before and after training and watch structure appear.
- **Delete the mask** (comment out the `masked_fill`). Training loss falls *absurdly* fast — the model reads the future, and predicting character `t+1` is easy when you can attend to it. Then generate: garbage. At generation time there *is* no future to peek at, so the model is lost without its crutch. Best single demonstration of why the mask exists — train/generate must live under the same rules.
- **Delete the scaling** (`* k.shape[-1]**-0.5`) and print `wei.var()` and a few softmaxed rows at init, ideally with a bigger `head_size` like 64: variance ~`head_size` instead of ~1, rows near one-hot. Training limps. Restore it, watch the rows soften.
- **Mask *after* softmax instead of before** (softmax first, then zero the future). Rows no longer sum to 1 — probability leaked to the future *before* you zeroed it, and what remains isn't a proper distribution over the past. Order matters: mask, then softmax.
- **`head_size=1`.** Each position gets a single scalar of routed information. Watch the loss barely beat bigram — capacity of the communication channel matters.

---

# 4.4 Step 4 — Multi-head attention + feed-forward

Two upgrades, both short, both principled.

## Multi-head: several conversations at once

One head computes one `(T, T)` listening pattern per input. But a character plausibly needs several *simultaneous* relationships: "what's the current word so far?", "am I inside a quotation?", "who's speaking?" — different queries against different keys. Multi-head attention is the embarrassingly simple fix: run **h independent heads in parallel, each smaller, and concatenate**:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)   # mix the heads' outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # concat on channel dim
        return self.dropout(self.proj(out))
```

The convention that keeps dimensions honest: **`head_size = n_embd // n_head`**, so the concatenation lands back at `n_embd` and blocks can stack without shape gymnastics. With `n_embd=384, n_head=6`, each head works in 64 dimensions. You're not adding capacity over one big head — you're *splitting* the same capacity into six independent listening patterns, and that diversity is what wins (break-it below). Two details:

- **`nn.ModuleList`, not a Python list** — a plain list hides the heads from PyTorch: their parameters wouldn't register, wouldn't move with `.to(device)`, wouldn't train. Silent and nasty; `ModuleList` exists precisely for this.
- **The output projection `proj`** gives the concatenated heads one linear layer to mix in — head 3's findings can modulate head 5's before the result rejoins the residual stream (Step 5). Every serious implementation has it.

(Efficiency footnote, for later: real implementations don't loop over head modules — they compute all heads in one batched matmul by reshaping `(B, T, n_embd)` into `(B, n_head, T, head_size)`, and then usually hand the whole thing to `F.scaled_dot_product_attention`, which fuses it into a FlashAttention kernel. Same math. Build the loop version — it's the one you can print your way through; take the fused one as a Phase 5 upgrade.)

## Feed-forward: after communicating, compute

Watch what the network has done so far: embed, then *route* (attention is weighted sums — linear all the way through, softmax notwithstanding, from the values' perspective). No token has actually **processed** what it gathered. The feed-forward network is that processing step — a small two-layer MLP applied to *each position independently* (no cross-token communication; that's attention's job):

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),     # expand 4x — the standard ratio
            nn.ReLU(),                          # (GPT-2 uses GELU; either is fine here)
            nn.Linear(4 * n_embd, n_embd),     # contract back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

It's literally Phase 2's MLP, applied at every `(b, t)` position with shared weights. The `4×` expansion is the transformer's conventional "thinking room" (from the original paper; nobody's found strong reasons to deviate at this scale), and roughly two-thirds of a transformer's parameters live in these MLPs — attention decides *what to look at*; the FFN does most of the *knowing*.

The rhythm to internalize, because Step 5 stacks it: **attention = communicate, feed-forward = compute.** Gather, then think. Repeat.

Wire both into the model — replace the single `sa_head` with `MultiHeadAttention(n_head, n_embd // n_head)` and add a `FeedForward(n_embd)` after it — then retrain: val loss **~2.4 → ~2.2**. The full file at the end of this step shows the two-line change in context. The ledger grows.

## The file so far — `gpt.py` after Step 4

Only the model section changes again. `Head` is untouched from Step 3; `MultiHeadAttention` and `FeedForward` are new; `OneHeadLM` becomes `MultiHeadLM` with two lines swapped in `forward`. The data, eval, and training sections are still Step 2's, unedited.

Since only the middle moved, here is the model section in full plus a one-line diff of the rest:

```python
# ---- config: two lines change ----
n_head   = 4                        # NEW  -> head_size = n_embd // n_head = 8
run_name = "runs/s4_mh_ff"          # CHANGED
# (batch_size 32, block_size 8, n_embd 32, dropout 0.0, lr 1e-3, max_iters 5000 — unchanged)

# ---------------- model ----------------
class Head(nn.Module):
    """One head of causal self-attention. UNCHANGED from Step 3 — copy it across verbatim."""
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k = self.query(x), self.key(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):                       # NEW
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)   # mix the heads' outputs
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)    # (B, T, num_heads*head_size)
        return self.dropout(self.proj(out))                    # (B, T, n_embd)

class FeedForward(nn.Module):                              # NEW
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)                                     # per-position, shared weights

class MultiHeadLM(nn.Module):                              # CHANGED (was OneHeadLM)
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)  # CHANGED: was Head(n_embd)
        self.ffwd = FeedForward(n_embd)                         # NEW
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        x = self.token_embedding(idx)      # (B, T, n_embd)
        x = self.sa(x)                     # communicate
        x = self.ffwd(x)                   # NEW: ...then compute
        logits = self.lm_head(x)           # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """UNCHANGED from Step 3 — copy it across verbatim."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---- training loop: one line changes ----
model = MultiHeadLM().to(device)          # CHANGED
```

Expected: **val ~2.4 → ~2.2**. If you get a shape error on `self.proj`, check that `n_embd % n_head == 0` — `32 // 4 = 8`, four heads concatenate back to 32, and the projection's input dimension is `num_heads * head_size`, which only equals `n_embd` when the division is exact.

### Break-it experiments — 4.4

- **One big head vs. many small (same total size):** `MultiHeadAttention(1, 32)` vs `MultiHeadAttention(4, 8)` — identical parameter budget, and the multi-head version trains to a better loss. Diversity of attention patterns, isolated as a variable.
- **Remove the feed-forward** and retrain: most of this step's gain vanishes. Communication without computation is just increasingly elaborate averaging.
- **Print per-head attention patterns** after training (`[h(x) for h in ...]` makes this easy — grab `wei` from each head for one input). Even half-trained, heads visibly differ: one near-diagonal (recent chars), one flatter (broad context). Nobody assigned those roles; they specialize because the gradient found it useful.
- **Swap ReLU for GELU** (`nn.GELU()`) — a wash at this scale, and now you know it's a *choice*, not a law. Real GPTs use GELU.

---

# 4.5 Step 5 — The Transformer block, stacked into a full GPT

Everything now exists; this step is assembly plus two ideas you've already met (residuals, normalization) and one you've been warned about (positions).

## The block — Phase 3.8, cashing its check

Naively stacking `attention → ffwd → attention → ffwd → ...` eight layers deep barely trains — you saw exactly this failure with plain deep CNNs in Phase 3.8, and the fix is the same fix: **residual connections**. Wrap each sublayer as `x = x + sublayer(x)`, so each block *refines* the representation instead of replacing it, and gradients get their highway to the bottom. Add the transformer's normalizer — **LayerNorm** — and you have the block:

```python
class Block(nn.Module):
    """Transformer block: communicate, then compute — both residual, both pre-normed."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))       # residual around attention
        x = x + self.ffwd(self.ln2(x))     # residual around feed-forward
        return x
```

Those two `x = x + ...` lines are the load-bearing structure of every LLM you've ever used. Two notes:

**LayerNorm vs. BatchNorm.** Same purpose as Phase 3's BatchNorm — keep activations well-scaled so depth stays trainable — but normalizing over the **feature dimension of each position independently** rather than across the batch. Why the switch for sequences: batch statistics are a mess when positions in a batch aren't comparable, and LayerNorm needs no running statistics, so it behaves *identically* in train and eval (one less `model.eval()` trap — dropout remains the one that bites). `nn.LayerNorm(n_embd)` normalizes each `(b, t)` vector to mean 0/var 1, then applies a learnable per-feature scale and shift — mechanically BatchNorm's cousin, exactly as the guide's Phase 3 table promised.

**Pre-norm.** Notice the norm goes *inside* the residual, before the sublayer — `x + sa(ln(x))` — not after the addition as in the original 2017 paper. Pre-norm keeps the residual stream itself untouched (a clean gradient path from loss to embedding) and trains more stably; it's what GPT-2 and essentially everything since uses. If you see `ln(x + sa(x))` in older diagrams, that's post-norm — know the distinction exists, then use pre-norm.

## The missing ingredient: position

Here's a fact worth proving to yourself (break-it below): **attention is permutation-blind.** Scores are pairwise dot products; nothing anywhere in Step 3–4 encodes *where* a token sits. Shuffle the input tokens and (mask effects aside) attention computes the same relationships. But "same three characters, different order" is the difference between `dog` and `god` — position is information, and we're currently throwing it away.

The fix is almost anticlimactic: a **second embedding table, indexed by position**. Token 7 gets `token_emb[char] + pos_emb[7]` — the vector entering block 1 encodes both *what* it is and *where* it is, and attention can learn to use either.

## The full model

```python
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)               # final norm before the head
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                    # (B, T, n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok + pos                                  # broadcasting: (B,T,C) + (T,C)
        x = self.blocks(x)                             # N rounds of communicate/compute
        x = self.ln_f(x)
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]            # NEW: crop context to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
```

Read it top to bottom and notice there is nothing you haven't built: two lookup tables, a stack of blocks made of attention + MLP + residuals + norms, a final norm, a linear classifier head. **That's a GPT.** GPT-2 is this same code with a BPE tokenizer, `n_layer=48, n_embd=1600`, `block_size=1024`, and a few init details. The architecture you can now debug line-by-line is, structurally, the real thing.

Three assembly details that cause real bugs:

- **`torch.arange(T, device=idx.device)`** — positions must be created on the model's device. Forget `device=` and you get the Phase 1 classic, `Expected all tensors to be on the same device`, now appearing mid-model.
- **`tok + pos` broadcasts** `(B, T, C) + (T, C)` — every batch row gets the same position vectors added. A Phase 1 broadcasting rep, now doing real work.
- **The `idx[:, -block_size:]` crop in `generate` now has two reasons behind it.** You already needed it in Step 3: `Head`'s `tril` buffer is a fixed `(block_size, block_size)` matrix, so a longer `T` makes `masked_fill` fail on shapes. Now the position table adds a second reason — it has exactly `block_size` rows, so an uncropped forward past the limit indexes `pos_emb[256]` and raises an index error (or, on GPU, a device-side assert). Step 2's bigram didn't care about length at all; every model from Step 3 on has a hard context window. That window — and the sliding crop — is exactly the "context length" you hear about in every LLM spec sheet, and why chat models forget the start of long conversations.

Run the small config (`n_embd=64, n_head=4, n_layer=4, block_size=32` or so) to verify it trains — val loss around **~2.0** territory — then move to Step 6 for the real run.

## The file so far — `gpt.py` after Step 5

This one is worth printing in full, because it is the first version that is *architecturally* a GPT — Step 6 only turns the dials. `Head`, `MultiHeadAttention`, and `FeedForward` arrive unchanged from Step 4; `Block` and `GPT` are new; the config grows an `n_layer` and a bigger `block_size`.

```python
"""
gpt.py — Step 5: transformer blocks (residual + pre-norm), position embeddings, real depth.
New vs Step 4: Block, positional embedding, ln_f, n_layer, temperature in generate.
Run: python gpt.py   ->  val ~2.2 becomes val ~2.0 on the small config.
"""
import os
import math
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config ----------------
batch_size    = 32
block_size    = 32          # CHANGED (was 8): depth is wasted on 8 characters of context
n_embd        = 64          # CHANGED (was 32)
n_head        = 4
n_layer       = 4           # NEW — how many Blocks to stack
dropout       = 0.0         # still 0: this model is too small to overfit 1MB
learning_rate = 1e-3        # Step 6 drops this to 3e-4 for the big config
max_iters     = 5000
eval_interval = 500
eval_iters    = 50
run_name      = "runs/s5_blocks"    # CHANGED

# ---------------- data ----------------  (unchanged from Step 1)
os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(url, "data/input.txt")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s):  return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i     : i+block_size    ] for i in ix])
    y = torch.stack([d[i + 1 : i+block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- model ----------------
class Head(nn.Module):                                     # unchanged since Step 3
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k = self.query(x), self.key(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):                       # unchanged since Step 4
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):                              # unchanged since Step 4
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):                                    # NEW
    """Communicate, then compute — both residual, both pre-normed."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))       # residual around attention
        x = x + self.ffwd(self.ln2(x))     # residual around feed-forward
        return x

class GPT(nn.Module):                                      # CHANGED (was MultiHeadLM)
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)          # NEW
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])  # NEW
        self.ln_f = nn.LayerNorm(n_embd)                                    # NEW
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)                                     # (B, T, n_embd)
        pos = self.position_embedding(torch.arange(T, device=idx.device))   # (T, n_embd)
        x = self.blocks(tok + pos)         # broadcast (B,T,C) + (T,C), then N blocks
        logits = self.lm_head(self.ln_f(x))                                 # (B,T,vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):   # temperature: NEW
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]        # now mandatory for TWO reasons: the
            logits, _ = self(idx_cond)             # head's tril AND the position table
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---------------- eval ----------------  (unchanged since Step 2)
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------- train ----------------  (unchanged since Step 2 except the class name)
if __name__ == "__main__":
    model = GPT().to(device)                                  # CHANGED
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(run_name)
    print(f"expected loss at init: {-math.log(1/vocab_size):.2f}")   # ~4.17, and this one hits it

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            writer.add_scalar("loss/train", losses["train"], it)
            writer.add_scalar("loss/val",   losses["val"],   it)

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.close()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, 500)[0].tolist()))
```

Compare this against §4.7's final file and the diff is **only the config block and the training harness** — same classes, same shapes, same everything. The architecture is finished here; Step 6 is scale plus the Phase 3 training machinery.

### Break-it experiments — 4.5

- **Ablate the residuals** (`x = self.sa(self.ln1(x))`, no `x +`) at `n_layer=6` and watch training crawl or stall — Phase 3.8's single-line ablation, reproduced on a transformer. Depth without highways doesn't train. Restore, and appreciate that the fix costs zero parameters.
- **Ablate the positional embedding** (`x = tok`). Two-part experiment: (1) loss gets *worse but not catastrophically* — the causal mask leaks some positional signal (position 0's row can only attend to itself...), a subtle and famous confound; (2) the *principle* is clean if you test attention directly — take a trained `Head`, feed `x` and a permuted `x[:, perm, :]` with the mask removed, and watch outputs match under the same permutation. Permutation-blindness, demonstrated.
- **Count parameters** (`sum(p.numel() for p in model.parameters())`). Predict the big contributors first, then check: at `n_embd=384, n_layer=6`, each block is ~1.8M (most of it feed-forward), embeddings ~0.12M, total ≈ **10.8M**. Compare: Phase 2 CNN 0.4M, Phase 3 ResNet 11M — your GPT is ResNet-sized. GPT-2 is 1,500× bigger; GPT-3, 16,000×. Same skeleton.
- **`ln_f` matters more than it looks:** remove the final LayerNorm and compare — usually a small but real degradation. The residual stream's scale grows with depth; the head appreciates a normalized input.
- **Feed a sequence longer than `block_size`** directly to `forward` and read the error you get. Now you know the failure signature before it ambushes you inside `generate`.

---

# 4.6 Step 6 — Train, generate, tune

## The config, and why each number

```python
# ---- the guide's 8GB-friendly config ----
block_size    = 256      # context: 256 chars ≈ 3-4 lines of a play
batch_size    = 32       # drop to 16 first if OOM
n_embd        = 384
n_head        = 6        # -> head_size 64, the standard ratio
n_layer       = 6
dropout       = 0.2      # NOW it earns its keep (10.8M params vs 1MB of text)
learning_rate = 3e-4     # the classic transformer LR; 1e-3 is too hot at this size
max_iters     = 5000
eval_interval = 250
eval_iters    = 100      # batches to average for the loss estimate
```

Why these numbers, in terms you now own:

- **`dropout=0.2` finally matters.** A 10.8M-parameter model versus a ~1M-character corpus is a Phase 3 overfitting setup on paper. You already placed dropout in three spots without comment — on the attention weights (`Head`), after the multi-head projection, and inside the feed-forward. That's the standard transformer placement; 0.2 is a sane dose for this data size.
- **`lr=3e-4`**: transformers are touchier than CNNs — the softmax in attention is a nonlinearity that misbehaves when updates are too big. `3e-4` with AdamW is the boring, correct default. (Real training adds LR *warmup* — a few hundred steps ramping up from 0 — before the cosine decay you know; optional here, standard at scale.)
- **`batch_size` vs `block_size` under 8 GB**: activation memory scales with `B × T` (and attention materializes `B × n_head × T × T` score matrices — quadratic in context). Your OOM ladder, in order: `batch_size` 32→16 (halves activations, only slows you down), then `block_size` 256→128 (quarters the attention matrices but genuinely shrinks the model's window), then `n_embd` (a smaller model — last resort). AMP, below, buys you the headroom to avoid the ladder entirely.

## Honest loss measurement — `estimate_loss`

You've been running this function since Step 2; here's why it's built the way it is. Single-batch loss is noisy (32 random chunks) and train-batch loss says nothing about generalization. The standard pattern — this phase's version of Phase 2's `evaluate` — averages many batches over both splits:

```python
@torch.no_grad()
def estimate_loss(model):
    model.eval()                       # dropout OFF for measurement — the 3.4 rule, still law
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out
```

Log both numbers to TensorBoard every `eval_interval`. You are reading the same two curves as Phase 3: val tracking train = healthy; val flattening while train keeps sinking = the 10.8M parameters starting to memorize Shakespeare — the diverging-curves picture, back for sequences. With `dropout=0.2` on this config, expect the gap to stay modest through 5000 iters.

## The training loop — Phase 3's harness, verbatim

Everything from 3.7 transplants: AdamW with weight decay, cosine schedule, AMP, best-checkpoint saving. One genuinely new tool: **gradient clipping**, standard for transformers because attention occasionally produces a freak large gradient that would yank the weights; clipping rescales any gradient whose overall norm exceeds 1.0. Note where it sits in the AMP sequence — this is exactly the `scaler.unscale_` pattern Phase 3.6 told you to file away:

```python
model = GPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
scaler = torch.amp.GradScaler("cuda")
writer = SummaryWriter("runs/s6_full")
best_val = float("inf")

for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        writer.add_scalar("loss/train", losses["train"], it)
        writer.add_scalar("loss/val",   losses["val"],   it)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], it)
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save(model.state_dict(), "checkpoints/gpt_best.pt")

    xb, yb = get_batch("train")
    with torch.amp.autocast("cuda"):
        _, loss = model(xb, yb)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)                                # un-scale grads first...
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # ...then clip at norm 1.0
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()                                          # per-ITERATION here (no epochs)
```

Note `scheduler.step()` runs **per iteration** with `T_max=max_iters` — the "no epochs anymore" bookkeeping from 4.1, reaching the scheduler. (Phase 3.5's warning about mixing per-epoch and per-batch stepping conventions is exactly the trap this sidesteps; the logged LR curve is still your proof it's right.)

Practicalities for the run itself: keep `nvidia-smi -l 2` in a second terminal — expect a few GB of VRAM with AMP on, and expect the Alienware's fans and thermal-throttled clocks on a run this long (Phase 0's warning, now relevant); elevate the chassis and let it work. A 5000-iteration run on the 3070 Ti is on the order of tens of minutes. Watch the first `estimate_loss` line before walking away: **~4.17 at iter 0**, as always.

## Expected results, and reading the output

By iter 5000, expect **val loss ≈ 1.5** (Karpathy's lecture lands at 1.48 with this architecture). Generate a long sample from the best checkpoint:

```python
model.load_state_dict(torch.load("checkpoints/gpt_best.pt"))
model.eval()
start = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(start, 2000)[0].tolist()))
```

You'll get something like:

```
DUKE VINCENTIO:
Well, your wit is in the care of side and that.

CLARENCE:
My lord, I have no other than the war
That thou dost say the crown'd of England's head...
```

Character names in caps with colons. Blank lines between speeches. Archaic near-words, iambic-ish rhythm, era-plausible vocabulary. **And it makes no sense.** Hold both facts at once: the *structure* is real, learned, and remarkable for a model this size predicting one character at a time — and the *meaning* isn't there, because meaning at this scale wasn't on offer. This is the guide's "expected, wonderful result." The gap between this output and a useful assistant is: ~4 more orders of magnitude of scale, a subword tokenizer, months of data curation, and post-training (instruction tuning, RLHF) — but **not a different architecture**. You built the architecture.

## Temperature — the creativity knob you already installed

`generate` divides logits by `temperature` before the softmax:

- **T < 1** (e.g. 0.8) sharpens the distribution — the model takes its favorite characters more often. Output gets more conservative, more repetitive, more "correct."
- **T = 1** samples the learned distribution as-is.
- **T > 1** flattens it — rarer characters get real probability. More surprising, then, past ~1.5, decaying into gibberish (you're re-approaching the uniform distribution — loss 4.17's ghost).

Try 0.5 / 0.8 / 1.0 / 1.3 on the same checkpoint and read a few hundred characters of each. This is the same `temperature` parameter in every LLM API, doing exactly this arithmetic. (The natural companion, **top-k sampling** — zero out all but the k most likely characters before sampling — is a 3-line addition with `torch.topk`; a good micro-exercise.)

## The file so far — `gpt.py` after Step 6

Step 6's complete file is **§4.7 below** — it's the last one, so it gets its own section. Before you jump there, here is the entire diff from Step 5, so you can apply it to your own file rather than pasting someone else's:

| Section | Step 5 | Step 6 |
|---|---|---|
| config | `block_size=32, n_embd=64, n_head=4, n_layer=4, dropout=0.0, lr=1e-3` | `block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2, lr=3e-4` + `weight_decay=0.1`, `run_name="runs/s6_full"` |
| data / model / eval | — | **completely unchanged.** Not one line of `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `GPT`, or `estimate_loss` differs. |
| training loop | AdamW, plain `loss.backward()` | + `weight_decay`, cosine scheduler stepped per iteration, AMP (`autocast` + `GradScaler`), `clip_grad_norm_` after `scaler.unscale_`, best-checkpoint saving |
| after training | generate 500 chars | reload the best checkpoint, `model.eval()`, generate 1000+ chars at a chosen temperature |

That's the honest summary of this step: **you are not writing new model code.** You're changing six numbers and transplanting the Phase 3 harness. Everything the extra val-loss points buy you comes from scale and training technique, on an architecture that was already correct at Step 5.

### Break-it experiments — 4.6

- **`temperature=0.05`.** Near-argmax: watch the model loop on its most confident patterns. Now you know *why* sampling, not argmax, was the right call in Step 2.
- **Trigger the OOM on purpose:** `batch_size=128` and watch `nvidia-smi` climb until CUDA gives up. Walk back down the ladder (batch → block → width) and note where it fits again. Better to meet this error deliberately than at iter 4000.
- **`dropout=0.0`, retrain, overlay the curves.** Val loss bottoms out earlier and *rises* while train keeps sinking — the cleanest overfitting picture the capstone can draw, and proof the 0.2 was earning rent.
- **Prompt it:** `idx = torch.tensor([encode("ROMEO:")], device=device)` instead of the zero token. The continuation picks up the cue — your first taste of prompting as *conditioning the sequence*, which is all prompting has ever been.
- **Train on your own text.** Swap `input.txt` for anything ≥ ~1MB (your notes, a public-domain book, source code). Watching it learn *your* corpus's texture is the phase's best reward — and code, with its rigid syntax, is eerily satisfying at char level.

---

# 4.7 The finished file — `gpt.py` after Step 6

The last "file so far": every piece from 4.1–4.6 in one runnable file, ~200 lines. Diff it against the Step 5 file above and only the config block and the training loop have moved.

```python
"""
gpt.py — a character-level GPT, built incrementally in Phase 4.
Structurally nanoGPT's lecture model + the Phase 3 training harness (AMP, cosine, clipping).
"""
import os
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config ----------------
block_size    = 256
batch_size    = 32
n_embd        = 384
n_head        = 6
n_layer       = 6
dropout       = 0.2
learning_rate = 3e-4
weight_decay  = 0.1
max_iters     = 5000
eval_interval = 250
eval_iters    = 100
run_name      = "runs/s6_full"

# ---------------- data ----------------
os.makedirs("data", exist_ok=True)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("data/input.txt"):
    urllib.request.urlretrieve(url, "data/input.txt")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
def encode(s):  return [stoi[c] for c in s]
def decode(ix): return "".join(itos[i] for i in ix)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i     : i+block_size    ] for i in ix])
    y = torch.stack([d[i + 1 : i+block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- model ----------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k = self.query(x), self.key(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding(idx)
        pos = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.blocks(tok + pos)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

# ---------------- eval ----------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ---------------- train ----------------
if __name__ == "__main__":
    model = GPT().to(device)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)
    scaler = torch.amp.GradScaler("cuda")
    writer = SummaryWriter(run_name)
    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model)
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}"
                  + ("  <- new best" if losses["val"] < best_val else ""))
            writer.add_scalar("loss/train", losses["train"], it)
            writer.add_scalar("loss/val",   losses["val"],   it)
            writer.add_scalar("lr", scheduler.get_last_lr()[0], it)
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(model.state_dict(), "checkpoints/gpt_best.pt")

        xb, yb = get_batch("train")
        with torch.amp.autocast("cuda"):
            _, loss = model(xb, yb)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    writer.close()
    print(f"best val loss: {best_val:.4f}")

    # ---------------- generate ----------------
    model.load_state_dict(torch.load("checkpoints/gpt_best.pt"))
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, 1000, temperature=0.9)[0].tolist()))
```

**The results ledger** — the arc of the phase in one table (your numbers will vary a bit; the *ordering* and *gaps* are what to verify):

| Run | What was added | Val loss ≈ | The lesson it carries |
|---|---|---|---|
| `s2_bigram` | embedding lookup only | ~2.5 | Pipeline works; architecture is the bottleneck, not training |
| `s3_onehead` | one attention head, `n_embd=32` | ~2.4 | Communication with the past helps |
| `s4_mh_ff` | multi-head + feed-forward | ~2.2 | Communicate *and* compute |
| `s5_blocks` | residual blocks + LN + pos emb (small config) | ~2.0 | Depth is trainable with highways |
| `s6_full` | scaled config + dropout + full harness | **~1.5** | Scale, on a sound skeleton |

✅ **commit:** `git commit -am "Phase 4: working character-level GPT"`

*(A note on `torch.compile`: nanoGPT uses it for a big speedup, but its support on native Windows is historically limited/absent — if you try it and hit Inductor errors, that's your platform, not your code. Skip it, or file it under "reasons to try WSL2 someday.")*

---

## Additions to the debugging table

Phase 4's new failure modes, in the appendix's format:

| Symptom | Usual cause | Fix |
|---|---|---|
| First loss far above ~4.9 (or ~0) | Reshape wrong in cross-entropy; logits/targets misaligned | Check `view(B*T, C)` / `view(B*T)`; print shapes. Reference: ~4.17 uniform, ~4.6–4.9 for the Step-2 bigram (`nn.Embedding` init), ~4.17 from Step 3 on |
| Loss ~4.17 and *never* moves | `y` not actually shifted; or generating logits from the wrong dim | Print a batch, verify the shift by eye |
| `IndexError` / device-side assert during `generate` | Sequence grew past `block_size`, indexed off the position table | The `idx[:, -block_size:]` crop |
| `Expected all tensors on same device` mid-forward | `torch.arange(T)` without `device=`; or mask as plain attribute | `device=idx.device`; `register_buffer` for the mask |
| Model trains but some layers never change | Heads in a plain Python list | `nn.ModuleList` |
| Train loss amazing, generated text garbage | Causal mask missing/wrong — model trained while peeking at the future | Verify `wei[0]` is lower-triangular |
| `wei` rows don't sum to 1 | Masked after softmax, or softmax over wrong dim | Mask with `-inf` *before* `softmax(dim=-1)` |
| Loss spikes/NaN mid-run | LR too hot for a transformer; or clipping applied to still-scaled grads | `lr=3e-4`; `scaler.unscale_` *before* `clip_grad_norm_` |
| Embedding layer errors on input | Indices are float, not `torch.long` | `dtype=torch.long` at encode time |
| Val loss rising while train falls | The 10.8M params memorizing 1MB of text | Dropout up, model down, or more data — Phase 3's playbook |

**The phase's golden debugging rule, extended:** when lost, print `.shape` and ask *which of B, T, C is each dimension?* Nearly every Phase 4 bug is a B/T/C mix-up, a missing mask, or a device mismatch.

---

## What Phase 4 actually taught you (the map to real LLMs)

Everything in the modern LLM stack now has a hook in your head:

| You built | In the real world |
|---|---|
| `encode`/`decode`, 65-char vocab | Tokenizers (BPE, ~50k–200k tokens) — Phase 5's first upgrade |
| `block_size=256` + the generate-time crop | "Context length 128k" headlines, and why models forget long chats |
| Next-char prediction on unlabeled text | Pretraining — the same objective, on trillions of tokens |
| `q @ kᵀ / √d`, mask, softmax, `@ v` | Scaled dot-product attention, verbatim, in every transformer paper |
| `x = x + sublayer(ln(x))` | The pre-norm residual stream — GPT-2 through today |
| `temperature` in `generate` | The same knob in every LLM API call |
| Prompting with `"ROMEO:"` | Prompting — conditioning the sequence, nothing more |
| Your 10.8M parameters | The same skeleton at 10⁹–10¹² parameters |

What you did *not* build — the honest list: subword tokenization, instruction tuning / RLHF (why chat models converse instead of just continuing text), KV-caching for fast generation (your `generate` recomputes everything each token — correct, just wasteful), FlashAttention-style fused kernels, and multi-GPU training. Every one of these is an *optimization or refinement of* the thing you built, not a different thing. That's the payoff of the capstone: the mystery is gone, and what remains is engineering.

---

## Suggested next moves

1. **Build it in the six-step order, recording val loss at each step.** The ledger is the proof-of-work — and if you skip to Step 6 and it's broken, you'll have no idea which of five ideas is the culprit. The steps *are* the debugger.
2. Do the three highest-value break-its if you do nothing else: **print the attention matrix** (4.3), **delete the causal mask and generate** (4.3), and **ablate the residuals at depth** (4.5). Each is one line and permanently reshapes your intuition.
3. Watch the Karpathy video *after* your Step 3 works, not before — his walkthrough of the masked-softmax trick will land completely differently once you've fought it yourself.
4. Run the full Step 6 config, beat val 1.6, and generate 2000 characters at temperature 0.8. Read them out loud. You earned it.
5. ✅ commit — then Phase 5's refactor (`data.py` / `model.py` / `train.py` / `generate.py`) is waiting, and this file's section boundaries are exactly the seams to cut along.
