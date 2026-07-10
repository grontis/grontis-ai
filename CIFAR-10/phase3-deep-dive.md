# Phase 3 Deep Dive — Going Deeper with CIFAR-10

Companion notes for **Phase 3** of `pytorch-char-gpt-guide.md`.
Detailed explanations + runnable examples for the phase where you stop learning *how to train* and start learning *how to train well* — overfitting, regularization, learning-rate scheduling, and your first real VRAM tool.

This picks up exactly where the Phase 2 deep dive left off. You own the whole classification pipeline: `Dataset` → `DataLoader` → `nn.Module` → `CrossEntropyLoss` → `train_one_epoch`/`evaluate` → checkpoints → TensorBoard. **None of that changes in Phase 3.** What changes is the *difficulty of the problem*, and difficulty is the teacher here: CIFAR-10 is hard enough that a naive model visibly overfits, which finally gives every regularization tool something real to do.

> **How to read this.** Phase 3 has a different rhythm than Phase 2. Phase 2 was "build A, then build B, compare." Phase 3 is **one experiment at a time**: establish a baseline, watch it fail in a specific way, add exactly one tool, and measure what that tool did. Resist the urge to turn everything on at once — if you add augmentation, dropout, weight decay, and a scheduler in one run, you'll get a better number and learn nothing about which knob did what. Every section below follows the pattern: *what it is → why it works → the code → what you should see in the curves*.

---

## The mental model for the whole phase

Phase 2's central number was test accuracy. Phase 3's central picture is **two curves diverging**:

```
accuracy
   │            train ────────────────────  99%+
   │           ╱
   │          ╱   ┌── the GAP = overfitting
   │         ╱    ▼
   │        ╱  ····································  test stalls ~84%
   │       ╱ ··
   │      ╱··
   │     ╱·
   └──────────────────────────────────────► epochs
```

The model keeps getting better at the training set — by **memorizing** it — while getting no better (or worse) at data it hasn't seen. Fashion-MNIST was too easy for this to bite hard: even the simple CNN generalized fine. CIFAR-10 is hard enough that your first model *will* produce this picture, and everything in this phase is a tool for closing that gap from a different angle:

| Tool | How it fights the gap |
|---|---|
| **Data augmentation** | Makes the training set effectively bigger — memorizing becomes impossible when every epoch shows slightly different images |
| **Dropout** | Randomly disables neurons during training so no single pathway can memorize |
| **Weight decay** | Penalizes large weights, biasing the model toward simpler functions |
| **LR scheduling** | Big steps early to explore, small steps late to settle — better final minima |
| **Mixed precision (AMP)** | Doesn't fight overfitting at all — it fights your **8 GB VRAM ceiling**, and it's the tool you'll lean on hardest in Phase 4 |

One habit before any code: **name every TensorBoard run after what you changed** (`runs/01_baseline`, `runs/02_augment`, `runs/03_aug_dropout`, ...). By the end of the phase you'll have a stack of overlaid curves that *is* the lesson — a visual record of each tool earning (or not earning) its place.

---

# 3.1 Data — CIFAR-10, color, and augmentation

## What CIFAR-10 actually is

CIFAR-10 is 60,000 **color** images, each **32×32 pixels**, across 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Split: **50,000 train / 10,000 test**. Same "10 balanced classes" shape as Fashion-MNIST — which is exactly why it's the right next step: your training code carries over unchanged, but the images are photographs of real objects at real angles under real lighting, and 32×32 is *blurry*. Pull one up and squint: some of these are hard for a human. Fashion-MNIST accuracy started at 88% and you fought for single points; CIFAR-10 starts much lower and the fight is the curriculum.

Two structural differences from Fashion-MNIST, and everything they touch:

1. **Three color channels instead of one.** Each image tensor is `(3, 32, 32)` — an RGB triple per pixel instead of one gray value. Code impact: the first `Conv2d` takes `in_channels=3`, and `Normalize` needs 3-element mean/std tuples. That's it — convs were built for multi-channel input from the start (your Phase 2 second conv already consumed 32 channels; 3 is nothing special).
2. **32×32 instead of 28×28.** New spatial arithmetic: three halvings give 32→16→8→4, pleasantly clean. Recompute your flatten size — don't carry `7*7` over by reflex (that's the #1 "port my Phase 2 model" bug).

```python
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.CIFAR10(root="data", train=True,  download=True, transform=ToTensor())
test_data  = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())

print(len(train_data), len(test_data))     # 50000 10000
img, label = train_data[0]
print(img.shape, img.dtype)                # torch.Size([3, 32, 32]) torch.float32
print(train_data.classes)                  # ['airplane', 'automobile', 'bird', ...]
```

The download is ~170 MB and lands in `data/` (already git-ignored). Look at a few images with the matplotlib snippet from Phase 2 — for color you plot `img.permute(1, 2, 0)` (matplotlib wants `(H, W, C)`, PyTorch stores `(C, H, W)`; `permute` reorders dimensions):

```python
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 6, figsize=(12, 2))
for i, ax in enumerate(axes):
    img, label = train_data[i]
    ax.imshow(img.permute(1, 2, 0))    # (3,32,32) -> (32,32,3) for display
    ax.set_title(train_data.classes[label])
    ax.axis("off")
plt.show()
```

## Normalization with three channels

Same idea as Phase 2, now with per-channel statistics. CIFAR-10's training-set mean and std are well known:

```python
from torchvision import transforms

normalize = transforms.Normalize(
    mean=(0.4914, 0.4822, 0.4465),   # per-channel mean (R, G, B)
    std=(0.2470, 0.2435, 0.2616),    # per-channel std
)
```

Don't take the numbers on faith — **compute them yourself once**, it's a good tensor exercise:

```python
import torch
from torch.utils.data import DataLoader

loader = DataLoader(train_data, batch_size=50000)   # the whole set in one batch
images, _ = next(iter(loader))                       # (50000, 3, 32, 32)
print(images.mean(dim=(0, 2, 3)))   # mean over batch, height, width -> one value per channel
print(images.std(dim=(0, 2, 3)))
```

`dim=(0, 2, 3)` says "reduce over every dimension *except* channels" — the kind of dimension-fluent move Phase 1 was building toward. You should recover the tuples above to ~3 decimal places.

## Data augmentation — the first real overfitting weapon

Here's the idea in one sentence: **apply random, label-preserving distortions to each training image every time it's fetched, so the model never sees the exact same input twice.**

A horizontally-flipped cat is still a cat. A cat shifted two pixels left is still a cat. But to a model trying to memorize pixel patterns, each variant is a *new image* — so the effective training set becomes vastly larger than 50,000, and memorization stops being a winning strategy. The model is forced to learn features that survive the distortions: shapes, textures, parts — the things that actually generalize. That's why augmentation is regularization, not just "more data."

The two standard CIFAR-10 augmentations from the guide:

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # pad to 40x40, crop a random 32x32 window
    transforms.RandomHorizontalFlip(),         # 50% chance of a mirror flip
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([          # NO augmentation — deterministic
    transforms.ToTensor(),
    normalize,
])
```

Unpack each piece:

- **`RandomCrop(32, padding=4)`** first pads the image with a 4-pixel border (making it 40×40), then cuts a random 32×32 window out of it. Net effect: the image content shifts by up to ±4 pixels in each direction, a different shift every epoch. This teaches translation robustness the honest way — by showing the model translated examples. (Note the order: geometric transforms operate on the PIL image, so they go *before* `ToTensor`; `Normalize` operates on tensors, so it goes after.)
- **`RandomHorizontalFlip()`** mirrors the image left-right with probability 0.5. Valid for CIFAR-10 because every class looks plausible mirrored (a mirrored truck is a truck). It would be *wrong* for digits — a mirrored "3" isn't a 3. Augmentation must preserve the label; that judgment call is yours per-dataset, and this is your first taste of it. (Vertical flips are also wrong here — CIFAR photos have a consistent up.)

### Two pipelines is the load-bearing decision

**Train gets augmentation; test never does.** The test set answers "how good is this model on real, unmodified images?" — randomly distorting it would make your accuracy number noisy and unrepresentative. Meanwhile normalization appears in *both* pipelines, because it's not augmentation — it's a deterministic preprocessing contract the model relies on for every input, forever (when you deploy a model, its normalization constants ship with it).

```python
train_data = datasets.CIFAR10("data", train=True,  download=True, transform=train_transform)
test_data  = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)
```

Batch size notes: 128 is the CIFAR-10 sweet spot on your 3070 Ti — these are tiny images and 8 GB is roomy here (VRAM pressure is a Phase 4 problem; this phase is where you learn the *tools* while they're cheap). Eval can use a bigger batch (256) since there are no gradients to store. The Windows `num_workers` caveat from Phase 2 still applies: start at `0`, and if you later add workers, keep everything under `if __name__ == "__main__":`.

One mechanical detail worth knowing: augmentation happens **on the fly, per fetch, on the CPU** — nothing is precomputed or stored. Epoch 1 and epoch 2 literally see different pixels for "the same" image index. That's also why heavier augmentation can bottleneck data loading — the first place `num_workers` starts earning its keep.

### Break-it experiments — 3.1

- **Fetch `train_data[0]` five times in a row** and print `img.sum()` each time. Different every fetch — that's augmentation working. Do the same for `test_data[0]`: identical every time.
- **Visualize the same training image 8 times** (subplots) and *see* the shifts and flips. To display normalized images, un-normalize first: `img * std_tensor + mean_tensor` (reshape the tuples to `(3,1,1)` — a broadcasting rep from Phase 1).
- **Put `RandomHorizontalFlip()` in the test transform** and run `evaluate` three times on the same checkpoint. Accuracy now jitters run-to-run — exactly why eval must be deterministic. Remove it.
- **Try `RandomVerticalFlip()`** in training and watch accuracy *drop* — an upside-down horse teaches the wrong invariance. Augmentation is a modeling decision, not free food.

---

# 3.2 The baseline — a deeper CNN (and BatchNorm)

## Methodology first: establish the failure

Before reaching for any Phase 3 tool, train a deeper CNN **without augmentation** (plain `ToTensor` + normalize on both splits) and log it to `runs/01_baseline`. You need to *see* the overfitting picture from the intro with your own eyes before the fixes will mean anything. This baseline is the control for every experiment in the phase.

## The architecture — three blocks of conv → BN → ReLU

Phase 2's CNN was two conv layers. CIFAR-10 wants more capacity, and the standard shape is **stacked blocks that alternate "extract features" (convs) with "shrink the map" (pooling)**, doubling the channel count each time the spatial size halves — trading resolution for feature richness as you go deeper:

```
input (B,3,32,32)
  ─► [Conv(3→64)   BN ReLU  Conv(64→64)   BN ReLU  MaxPool]  ─► (B,64,16,16)
  ─► [Conv(64→128) BN ReLU  Conv(128→128) BN ReLU  MaxPool]  ─► (B,128,8,8)
  ─► [Conv(128→256)BN ReLU  Conv(256→256) BN ReLU  MaxPool]  ─► (B,256,4,4)
  ─► Flatten (4096) ─► Linear(4096→512) ─► ReLU ─► Linear(512→10)
```

```python
import torch.nn as nn

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64),      # (B,3,32,32)  -> (B,64,16,16)
            conv_block(64, 128),    #              -> (B,128,8,8)
            conv_block(128, 256),   #              -> (B,256,4,4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # -> (B, 256*4*4 = 4096)
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),     # -> (B,10) logits
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

Notice what you *didn't* have to relearn: it's still `nn.Module`, still ends in bare logits, still pairs with `CrossEntropyLoss` and the unchanged `train_one_epoch`/`evaluate`. Two genuinely new things:

- **A block-builder function.** Writing `conv_block` once and calling it three times beats copy-pasting nine layers — and "define a block, stack it N times" is *exactly* how the Phase 4 transformer is built. You're practicing the pattern early.
- **`nn.BatchNorm2d`** — new layer, worth a full section.

## `nn.BatchNorm2d` — what it is and why deep nets want it

Batch normalization re-standardizes each channel's activations — for every channel, across the batch and all spatial positions, it subtracts the mean and divides by the std, then applies a learnable per-channel scale and shift (so the network can undo it where that helps). `BatchNorm2d(64)` normalizes 64 channels and holds 64 scales + 64 shifts as learnable parameters.

Why bother? You already normalized the *input*. But after a few layers of matmuls and ReLUs, the *intermediate* activations drift — some channels running hot, some near-dead — and each layer has to keep re-adapting to the shifting distribution coming out of the layer below. BatchNorm re-centers the signal at every block, which in practice means: **you can train deeper nets, at higher learning rates, that converge faster and a bit more stably.** It also adds a mild regularization side-effect (each image's normalization depends on which batch-mates it was shuffled with — a little noise, for free). It's not magic and it's not always used (Phase 4's transformer uses LayerNorm, its sequence-friendly cousin) — but for CNNs it's the standard, and the deeper net trains noticeably better with it. Prove that to yourself in the break-it experiments.

**The trap that makes `model.train()`/`model.eval()` non-negotiable from here on:** BatchNorm behaves *differently* in the two modes. During training it normalizes using the **current batch's** statistics and quietly maintains running averages of them; during eval it uses those **stored running averages** instead. Why the switch matters: at inference you might feed one image at a time — a "batch" of 1 has no meaningful statistics — and eval results must not depend on what else happened to be in the batch. Phase 2 built the `train()`/`eval()` habit when it changed nothing; this is the phase where forgetting it produces the appendix's "eval accuracy weirdly low" bug for real. (Dropout, next section, is the second layer with split personalities.)

## Run the baseline and stare at the failure

Train ~30 epochs with `Adam(lr=1e-3)`, logging train **and** test curves to TensorBoard (`runs/01_baseline`). Expected shape of the result on your machine (numbers will vary a couple points):

- Train accuracy marches to **99%+** and pins there.
- Test accuracy climbs fast, then **stalls somewhere in the low-to-mid 80s** by epoch ~10–15 and stops improving — or slides backwards while **test *loss* rises**.
- That ~15-point gap is the enemy. Screenshot it; every following section is measured against this picture.

The test-loss detail deserves a beat: it's common to see test *accuracy* plateau while test *loss* climbs. The model still picks the same (often right) class, but grows ever more confident on the wrong ones — memorization curdling into miscalibration. **Rising eval loss is your earliest overfitting alarm**, usually firing before accuracy visibly degrades. Log both; read both.

### Break-it experiments — 3.2

- **Delete the `BatchNorm2d` lines** and retrain with the same lr. Slower convergence, likely a lower plateau, possibly instability. Restore them. (While you're at it: with BN removed, crank `lr` to `3e-3` and watch it struggle; with BN restored, the same lr is often fine — "train at higher learning rates," demonstrated.)
- **Recompute the flatten arithmetic** blind: kernel 3 / padding 1 / three pools on a 32×32 input. Confirm `256*4*4`. Then run the Phase 2 dummy-tensor trick to check yourself.
- **Count parameters** (`sum(p.numel() for p in model.parameters())`) — ~4.7 M, versus ~0.4 M for the Phase 2 CNN. More capacity is *why* it can memorize 50k images; capacity and overfitting arrive together.
- **Overfit a single batch on purpose:** train on just one batch of 128 for 200 steps and confirm the model reaches 100% on it. This classic sanity check proves the pipeline can learn *at all* — if it can't even memorize 128 images, you have a bug, not an overfitting problem.

---

# 3.3 Reading the curves — diagnosis before treatment

You now have a baseline run in TensorBoard. Before adding tools, learn to read the four curve shapes you'll see for the rest of your life:

| What you see | Diagnosis | Response |
|---|---|---|
| Train acc high & climbing, test acc stalled/dropping (gap growing) | **Overfitting** | Augmentation, dropout, weight decay — this phase's toolkit |
| Train acc *itself* low and flat | **Underfitting** — model too weak or lr wrong | Bigger model, higher lr, train longer; regularization would make it *worse* |
| Both curves still climbing together at the end | **Undertrained** | Just train more epochs |
| Loss spikes to huge values / `nan` | **Instability** | Lower lr; check data; (in AMP, see 3.6) |

The order of operations matters: **diagnose, then treat.** Beginners reflexively add dropout to an underfitting model — the model gets *worse*, because regularization deliberately handicaps training, and a model that can't even fit the training set doesn't need handicapping. The train curve tells you which regime you're in: **train accuracy near-perfect = the model has capacity to spare = regularization has something to trade away.** Your baseline is firmly in that regime, so proceed.

A practical TensorBoard workflow for this phase:

- One subfolder per experiment: `runs/01_baseline`, `runs/02_augment`, `runs/03_aug_wd`, ... TensorBoard overlays all of them; toggle runs with the checkboxes on the left.
- Log four scalars per epoch (`loss/train`, `loss/test`, `acc/train`, `acc/test`) exactly as in Phase 2, plus — new this phase — the learning rate (see 3.5).
- Keep a plain-text lab notebook (`experiments.md`): one line per run — what changed, final test acc, and what you conclude. Fifteen runs in, you will not remember what `runs/07` was. This habit *is* "software development in the AI space."

**Also fix your checkpoint criterion now:** keep saving best-by-test-accuracy (Phase 2's pattern), and note that from this phase on, "best epoch" and "last epoch" genuinely differ — the best checkpoint often lands mid-run, before late-stage overfitting erodes it. The save-the-best pattern stops being a nicety and starts earning its keep.

> **Honest-measurement footnote.** We're using the test set to pick checkpoints and steer decisions, which technically leaks information — done rigorously you'd carve a *validation* split (e.g. 45k/5k via `torch.utils.data.random_split`) for all decisions and touch the test set once, at the very end. For a learning project, using test-as-validation is standard and fine; just know the distinction exists, because in any real project it's load-bearing. The guide's "validation" curves and our "test" curves are the same concept.

---

# 3.4 Regularization — Dropout and weight decay

Augmentation attacks overfitting through the *data*. These two attack it through the *model*. Add them one at a time, each as its own named run.

## First: turn on augmentation and measure it

Swap in the `train_transform` from 3.1 (this is the one-line change: the transform argument), log to `runs/02_augment`, retrain. Expect:

- Train accuracy **no longer pins at 99%** — it might top out around 92–95%. This is not a problem; it's the mechanism. The training task got legitimately harder, so the memorization score dropped.
- Test accuracy climbs **higher** than baseline — typically into the **high 80s** — and keeps improving for more epochs before stalling. The gap narrows from both ends.

Burn this asymmetry in: **train accuracy went down, test accuracy went up, and only the second number matters.** From now on, train accuracy is a diagnostic, not a goal.

## `nn.Dropout` — randomly break the network so it learns redundancy

During training, dropout zeroes each element of its input independently with probability `p` (and scales survivors by `1/(1-p)` so the expected magnitude is unchanged). Every forward pass, a different random subset of neurons is silenced.

Why that helps: without dropout, a network can route its answer through a few fragile, hyper-specific pathways — co-adapted neurons that only work as a clique, which is what memorization looks like structurally. With dropout, any neuron might vanish at any moment, so the network is forced to spread evidence across **redundant, independently-useful features**. It's ensemble thinking baked into one model: you're effectively training a huge family of thinned sub-networks that share weights.

At eval time dropout **turns off completely** — full network, no zeroing, deterministic output (the `1/(1-p)` train-time scaling is what makes this seamless). This is the second layer where `model.train()`/`model.eval()` changes behavior, and forgetting `eval()` now costs you real accuracy: you'd be randomly deleting half the classifier's neurons *while measuring it*.

Placement and dosage for this net — dropout is most at home on the **wide linear layers**, which hold most of the memorization capacity:

```python
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.ReLU(),
    nn.Dropout(0.5),               # <- the classic placement: after activation, before final layer
    nn.Linear(512, 10),
)
```

`p=0.5` is the traditional default for linear layers. In the conv stack, plain per-element dropout is less effective (neighboring pixels are correlated, so zeroing scattered pixels barely removes information — `nn.Dropout2d`, which drops entire channels, exists for that job), and BatchNorm already contributes its own noise there. Start with the single classifier dropout; treat conv-stack dropout as an experiment, not a default.

Run it as `runs/03_aug_dropout`. Expect a modest bump — augmentation already did the heavy lifting, and tools that fight the same enemy overlap. That's a real lesson too: **regularizers don't stack linearly.**

## Weight decay — a gentle pull toward zero

Weight decay adds a second objective alongside the loss: **keep the weights small.** Every step, each weight is nudged toward zero by a tiny amount proportional to its size (equivalent to penalizing `Σw²` — "L2 regularization"). To keep a weight large, the data has to keep re-earning it through gradients; weights that only exist to memorize quirks of specific training images don't get enough consistent signal to pay that rent, and decay erodes them. Geometrically: smaller weights → smoother, less contorted decision boundaries → better generalization.

One line in the optimizer:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
```

**Note the switch from `Adam` to `AdamW`.** Both accept `weight_decay`, but plain `Adam` implements it by adding the penalty into the gradient, where Adam's per-parameter adaptive scaling then distorts it — parameters with large gradient history get almost no decay. `AdamW` (the W is literally for "weight decay") applies the decay directly to the weights, *decoupled* from the gradient machinery, which is both more correct and empirically better. It's the standard modern choice — and the optimizer you'll use for the GPT in Phase 4, so switch now and don't look back. Everything else about the optimizer interface is identical (the Phase 2 lesson: `zero_grad`/`step` don't change a character).

Dosage intuition: `weight_decay` for AdamW typically lives in `1e-2`–`1e-1` (versus `1e-4`-ish for the coupled version in plain Adam/SGD — the scales aren't comparable, so don't port numbers between them). Too much decay and you'll see the *underfitting* signature — train accuracy sagging — which by 3.3 you now know how to recognize and respond to (back it off). Run as `runs/04_aug_dropout_wd`.

One convention you'll meet in the wild (fine to skip for now): decay is usually applied only to weight *matrices*, not to biases or BatchNorm/LayerNorm scale parameters. PyTorch's one-liner decays everything; the difference is small at this scale. Phase 4's optimizer setup does it properly with parameter groups — file the term away.

### Break-it experiments — 3.4

- **`Dropout(0.9)`.** Train accuracy craters — you've lobotomized the classifier. The underfitting picture from 3.3, self-inflicted. Walk `p` through 0.2 / 0.5 / 0.7 and find the sweet spot yourself.
- **Comment out `model.eval()` in `evaluate`** (temporarily!) with dropout in the model. Watch test accuracy drop by points and *jitter between identical calls* — the appendix bug row, experienced firsthand. Restore it and re-memorize the rule.
- **`weight_decay=1.0`.** Weights can't hold on to anything; train accuracy sags visibly. Feel the underfit, then step back down.
- **Print the model's total weight magnitude** (`sum(p.pow(2).sum() for p in model.parameters()).sqrt()`) at the end of a run with and without decay. Smaller with — the penalty is real, not metaphorical.

---

# 3.5 Learning-rate scheduling — `CosineAnnealingLR`

## Why one fixed learning rate is leaving accuracy on the table

The learning rate that's right at epoch 1 is wrong at epoch 45. Early in training, weights are random and far from good — you want **big steps**. Late in training, the model sits near a good minimum and a big step just bounces it back and forth *across* the bowl instead of settling **into** it. A fixed `1e-3` forever means your final weights are still jittering at a radius proportional to the lr — you never actually converge, you orbit.

The fix: **decay the learning rate over training.** A scheduler adjusts the optimizer's lr on a fixed program. Cosine annealing — the guide's pick, and the one Phase 4 reuses — glides from the initial lr down to ~0 along a half-cosine:

```
lr
1e-3 ┤────╮
     │     ╲          smooth: fast-ish early decay,
     │      ╲___      long gentle landing
     │          ╲──╮
   0 └──────────────╰──► epoch (T_max)
```

## The code — three touches

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

for epoch in range(num_epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
    scheduler.step()                                            # <- once per epoch, AFTER training

    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)  # <- log it; always log it
    # ... the four loss/acc scalars as before ...
```

- **`T_max=num_epochs`** is the length of the glide — the lr hits ~0 exactly when training ends. This quietly couples your schedule to your run length: a 30-epoch cosine run and a 60-epoch one are *different experiments*, and stopping a cosine run early means you never got the low-lr phase where much of the final accuracy materializes. Decide your epoch budget before the run.
- **`scheduler.step()` once per epoch, after the epoch's training.** The scheduler wraps the optimizer and rewrites its lr; the optimizer never knows anything happened. (Some schedulers step per *batch*; cosine-by-epoch is the convention here. Mixing the two conventions up — stepping an epoch-scheduler every batch — silently burns through the whole schedule in the first epoch. Logging the lr curve is how you'd catch it.)
- **`scheduler.get_last_lr()[0]`** reads the current lr (a list — one entry per parameter group; you have one). The logged curve should be a clean half-cosine. If it isn't, you found a bug in seconds instead of never.

What to expect in `runs/05_full` (all tools on): the signature cosine shape is test accuracy **ticking up noticeably in the final third** of training as the lr gets small — runs that looked plateaued suddenly find another point or two. With augmentation + dropout + weight decay + cosine over ~50 epochs, this architecture should land around **90–92%** test accuracy. Compare against `runs/01_baseline`'s low-80s stall: that spread is Phase 3, drawn as curves.

(Two relatives you'll meet later, no need to implement now: **warmup** — a few epochs ramping the lr *up* from 0 before decaying, standard for transformer training in Phase 4 — and `ReduceLROnPlateau`, which reacts to a stalling metric instead of following a fixed program.)

### Break-it experiments — 3.5

- **Set `T_max=5` on a 50-epoch run.** The lr hits ~0 by epoch 5 and then... cosine *rises again* (it's a cosine — it cycles). Watch the lr plot do a full wave and connect the wobble in your loss curve to it.
- **Same run, no scheduler,** overlaid in TensorBoard. Usually within a point or two — but look at *when* each run got there and how noisy the last ten epochs are. The scheduler buys the calm ending.
- **Call `scheduler.step()` inside the batch loop** (wrongly) and look at the logged lr: the schedule evaporates within epoch 1. Now you know the failure smell.

---

# 3.6 Mixed precision (AMP) — your first VRAM tool

## Why this exists

Everything so far ran in `float32` — 4 bytes per number, for every weight, activation, and gradient. Your RTX 3070 Ti has hardware (tensor cores) that runs `float16` math **dramatically faster**, and halving the bytes also halves most of the memory bill. CIFAR-10 doesn't *need* the savings — that's precisely why now is the time to learn the tool. In Phase 4 the guide's GPT config is sized for 8 GB *with AMP on*; you want the mechanics boring and familiar before they're load-bearing.

The catch that makes this an API and not a flag: `float16` has a narrow range. Very small values **underflow to zero** — and gradients are full of very small values. Train naively in fp16 and quiet layers silently stop learning. So PyTorch's *automatic* mixed precision does two things:

1. **`autocast`** — runs the forward pass with per-op precision choices: matmuls and convolutions (the speed wins) in fp16, range-sensitive ops (like the softmax inside your loss) kept in fp32. You write no casts; it's a context manager that intercepts ops.
2. **`GradScaler`** — protects the backward pass from underflow with a trick: multiply the loss by a large factor (say 65536) *before* `backward()`. By the chain rule every gradient comes out scaled by the same factor — lifted safely out of underflow territory — then the scaler divides them back down inside `step()` before the optimizer uses them. Mathematically a no-op; numerically a rescue. The scaler also auto-tunes its factor: if scaled gradients overflow to `inf`, it **skips that optimizer step**, halves the factor, and retries — occasional "skipping step" behavior early in training is normal, not a bug.

## The code — the five-step skeleton, annotated

This is the guide's snippet, placed in your actual `train_one_epoch`:

```python
scaler = torch.amp.GradScaler("cuda")        # create ONCE, outside the loops (it has state)

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):       # forward + loss under autocast
            preds = model(images)              # convs/matmuls run in fp16
            loss = loss_fn(preds, labels)      # loss internals stay fp32

        optimizer.zero_grad()
        scaler.scale(loss).backward()          # backward on the SCALED loss
        scaler.step(optimizer)                 # unscale grads, then optimizer.step()
                                               #   (auto-skipped if grads overflowed)
        scaler.update()                        # adapt the scale factor for next time

        running_loss += loss.item() * images.size(0)          # bookkeeping: unchanged
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total
```

Map it onto the sacred five moves: forward and loss now live under `autocast`; `backward` becomes `scaler.scale(loss).backward()`; `optimizer.step()` becomes `scaler.step(optimizer)` + `scaler.update()`. `zero_grad`, the bookkeeping, the scheduler, checkpointing — untouched. **`evaluate` needs none of this** (no gradients → nothing to underflow → no scaler; you *may* wrap eval's forward in plain `autocast` for speed, but it's optional).

The rules that prevent the classic AMP bugs:

- **One `GradScaler` for the whole run**, created next to the optimizer — not per-epoch, not per-batch. It's stateful; recreating it resets its calibration.
- **Only `backward()` sees the scaled loss.** Your logged `loss.item()` comes from the original `loss` tensor — unscaled, directly comparable to your fp32 runs.
- **Don't touch gradients between `backward()` and `scaler.step()`** without unscaling first — they're still multiplied by the factor. (This matters the day you add gradient clipping, which Phase 4 will: the pattern is `scaler.unscale_(optimizer)` → clip → `scaler.step(optimizer)`. File it away.)
- The old spelling `torch.cuda.amp.autocast()` litters tutorials everywhere; `torch.amp.autocast("cuda")` is the current API. Same machinery.

## Verify it, like an engineer

Run the identical experiment as `runs/06_full_amp` and check three things:

1. **Speed** — time your epochs (`time.perf_counter()` around the call). Expect a solid speedup on the 3070 Ti; small models see less than big ones because data loading and Python overhead don't shrink, only the GPU math does. (Phase 4's transformer, which is almost pure matmul, benefits much more.)
2. **Memory** — watch `nvidia-smi -l 2` during an fp32 run vs an AMP run. Lower peak usage. This delta is your Phase 4 headroom.
3. **Accuracy** — final test accuracy within normal run-to-run noise of the fp32 run. AMP done right costs essentially nothing; that's the whole point of the scaler machinery.

That three-way check — faster? smaller? still correct? — is the template for evaluating *any* performance tool, and doing it here on a small model is what makes you trust AMP when the GPT is on the line.

### Break-it experiments — 3.6

- **Remove the scaler** (keep `autocast`, call plain `loss.backward()` / `optimizer.step()`). It may even *seem* fine on this small model — which is the scary part; underflow is silent and shows up as mysteriously mediocre convergence on harder problems. Understand why the demo is undramatic here, and use the scaler anyway.
- **Print dtypes under autocast:** inside the block, `print(preds.dtype)` → `torch.float16`; print a weight's dtype → still `torch.float32`. Autocast changes *computation* precision, not your stored parameters — the master weights stay fp32. That asymmetry is the "mixed" in mixed precision.
- **Print `scaler.get_scale()` every 100 steps** for the first epoch. Watch it calibrate — possibly halving once or twice early, then stabilizing. The machinery, visible.

---

# 3.7 Putting it all together — the full Phase 3 script

Everything from 3.1–3.6 in one runnable file. This is the "final form" of the classifier project — and, structurally, about 90% of the Phase 4 training harness with a different model plugged in.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import os, time

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# ---------------- config (one obvious place; a dict for now, a dataclass in Phase 5) ----
cfg = dict(
    batch_size=128, epochs=50, lr=1e-3, weight_decay=5e-2,
    run_name="runs/06_full_amp",
)

# ---------------- data ----------------
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([transforms.ToTensor(), normalize])

train_data = datasets.CIFAR10("data", train=True,  download=True, transform=train_transform)
test_data  = datasets.CIFAR10("data", train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

# ---------------- model ----------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(),
        nn.MaxPool2d(2),
    )

class DeepCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64), conv_block(64, 128), conv_block(128, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = DeepCNN().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
scaler = torch.amp.GradScaler("cuda")

# ---------------- train / eval ----------------
def train_one_epoch(model, loader):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda"):
            preds = model(images)
            loss = loss_fn(preds, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            running_loss += loss.item() * images.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# ---------------- the loop ----------------
if __name__ == "__main__":
    writer = SummaryWriter(cfg["run_name"])
    os.makedirs("checkpoints", exist_ok=True)
    best_acc = 0.0

    for epoch in range(cfg["epochs"]):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader)
        te_loss, te_acc = evaluate(model, test_loader)
        scheduler.step()

        writer.add_scalar("loss/train", tr_loss, epoch)
        writer.add_scalar("loss/test",  te_loss, epoch)
        writer.add_scalar("acc/train",  tr_acc,  epoch)
        writer.add_scalar("acc/test",   te_acc,  epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), "checkpoints/cifar_best.pt")

        print(f"epoch {epoch+1:3d} | train {tr_acc:.3f} | test {te_acc:.3f} "
              f"| lr {scheduler.get_last_lr()[0]:.5f} | {time.perf_counter()-t0:.1f}s"
              + ("  <- new best" if te_acc == best_acc else ""))

    writer.close()
    print(f"best test accuracy: {best_acc:.3f}")
```

**Expected results ledger** — the arc of the whole phase in one table (your exact numbers will differ by a point or two; the *ordering* and the *gaps* are what to verify):

| Run | Setup | Train acc | Test acc | The lesson it carries |
|---|---|---|---|---|
| `01_baseline` | Deep CNN, no tools | 99%+ | ~82–85%, stalls early | Capacity memorizes; see the gap |
| `02_augment` | + crop/flip | ~92–95% | ~87–89% | Train acc down, test acc up — good |
| `03/04` | + dropout, + weight decay | ~90–94% | +0.5–1.5% more | Regularizers overlap; smaller returns |
| `05_full` | + cosine schedule, 50 epochs | — | **~90–92%** | The late-run low-lr gains |
| `06_full_amp` | same, + AMP | — | same ± noise | Faster, smaller, still correct |

✅ **commit:** `git commit -am "Phase 3: CIFAR-10 CNN with augmentation, AMP, scheduling"`

---

# 3.8 Stretch goal — a small ResNet (residual connections)

## The problem residuals solve

The obvious next move after "deeper helped" is "even deeper" — and it fails in a surprising way. Stack enough plain conv layers and the *training* accuracy gets **worse** than the shallower net's. Not overfitting (that's a *test*-side failure) — the deeper net can't even fit the training data as well, despite strictly more capacity. Two things go wrong: gradients degrade as they're chained backward through dozens of layers, and — more fundamentally — a deep stack struggles to learn even the *identity* function, so extra layers can't gracefully get out of the way when they're not needed.

The 2015 ResNet insight is disarmingly simple. Instead of asking a block to produce the whole output, let it produce a **correction to its input**:

```
plain block:      x ──► [conv → bn → relu → conv → bn] ──► out
residual block:   x ──► [conv → bn → relu → conv → bn] ──►(+)──► relu ──► out
                   └────────────── skip ──────────────────┘
```

`out = F(x) + x`. The consequences, in order of importance:

- **Doing nothing is now free.** If a block's weights are ~0, the block is the identity — the signal passes through untouched. Extra depth can never hurt expressiveness, so very deep networks become trainable.
- **Gradients get a highway.** In the backward pass, the `+ x` branch delivers gradient *directly* to earlier layers, bypassing the conv stack. The chain of degradation is broken — every layer hears the loss clearly, no matter how deep it sits.
- **Learning is reframed as refinement.** Each block nudges a running representation rather than rebuilding it from scratch — stack many small refinements and depth finally pays.

**Why this is in the guide at all:** Phase 4's transformer block is `x = x + attention(x)` then `x = x + feedforward(x)` — residual connections are *the* structural trick that makes it trainable, reused verbatim. Build the intuition here, on a model you can train in minutes, and the transformer's skeleton will look familiar instead of arbitrary.

## The residual block, in code

One wrinkle the diagram hides: `F(x) + x` requires the two shapes to match. When a block changes the channel count or downsamples (stride 2), the skip path must be transformed too — the standard fix is a 1×1 conv on the shortcut (a per-pixel linear remap of channels; no spatial mixing, minimal parameters):

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()          # identity, when shapes already match
        if stride != 1 or in_ch != out_ch:       # otherwise: 1x1 conv to match them
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)             # <- THE line. Everything else is scaffolding.
        return torch.relu(out)
```

Details worth noticing:

- **`bias=False` on convs followed by BatchNorm** — BN's learnable shift immediately replaces any bias, so the bias parameters would be dead weight. A standard idiom you'll see in every ResNet implementation.
- **Downsampling by `stride=2` instead of MaxPool** — the first conv of a stage-opening block strides, halving H×W while its shortcut's 1×1 conv strides in lockstep. Just a different way to shrink the map; ResNets happen to do it this way.
- **The skip addition comes *before* the final ReLU** — the block's output stays able to represent the clean identity path plus a correction.

## Stacking blocks into a small CIFAR ResNet

This is the classic CIFAR-style ResNet layout (channels 64→128→256→512, two blocks per stage — essentially ResNet-18 adapted to 32×32 inputs):

```python
class SmallResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(                       # gentle entry: no downsampling yet
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(ResidualBlock(64, 64),              ResidualBlock(64, 64))     # 32x32
        self.stage2 = nn.Sequential(ResidualBlock(64, 128, stride=2),   ResidualBlock(128, 128))   # 16x16
        self.stage3 = nn.Sequential(ResidualBlock(128, 256, stride=2),  ResidualBlock(256, 256))   # 8x8
        self.stage4 = nn.Sequential(ResidualBlock(256, 512, stride=2),  ResidualBlock(512, 512))   # 4x4
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                     # (B,512,4,4) -> (B,512,1,1): mean over space
            nn.Flatten(),                                # -> (B,512)
            nn.Linear(512, num_classes),                 # -> (B,10) logits
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.head(x)
```

New in the head: **`AdaptiveAvgPool2d(1)`** simply averages each channel over all spatial positions ("how strongly did feature #k fire *anywhere*?"), replacing the big `Flatten → Linear(4096, 512)` block — and with it, most of the classifier's memorization-prone parameters. This is why ResNets typically don't need the dropout layer.

Drop it into the 3.7 script — `model = SmallResNet().to(device)`, everything else identical (the model-agnostic skeleton, still paying off). It's ~11 M parameters and noticeably slower per epoch than `DeepCNN`, which makes AMP genuinely welcome. With the full recipe over 50 epochs, expect **~93–95%** test accuracy — a clear step past the plain deep CNN, and your empirical proof that residual connections buy trainable depth.

### Break-it experiments — 3.8

- **Ablate the skip:** change the addition line to `out = out` (keep everything else), retrain, and watch training itself struggle — slower convergence, worse final train *and* test accuracy. This is the cleanest single-line ablation in deep learning; you just reproduced the ResNet paper's core figure in miniature.
- **Verify identity-friendliness:** construct a `ResidualBlock(64, 64)`, zero all its parameters (`for p in block.parameters(): torch.nn.init.zeros_(p)`), set `.eval()`, and confirm `block(x)` equals `relu(x)` — with dead weights the block collapses to (almost) a pass-through. A plain conv block with zeroed weights outputs *zeros*, destroying the signal. That asymmetry is the whole idea.
- **Go deeper for free:** double the blocks per stage (2→4). It still trains cleanly — depth without drama. Try to imagine 16 plain conv blocks doing the same. Watch VRAM in `nvidia-smi` while you're at it; deeper = more activations stored for backward.

---

## Additions to the debugging table

Phase 3 introduces new failure modes; append these to the appendix's table in your head:

| Symptom | Usual cause | Fix |
|---|---|---|
| Eval accuracy jitters between identical runs | Augmentation in the test transform, or missing `model.eval()` with dropout | Deterministic test pipeline; audit `train()`/`eval()` calls |
| Eval accuracy weirdly low (for real this time) | Missing `model.eval()` — dropout still firing / BN using batch stats | Call `model.eval()` in `evaluate`, always |
| Train acc *down* after adding a tool | That's regularization working — or overdosed | Compare test acc; if test also fell, reduce `p`/`weight_decay` |
| Loss curve wobbles on a cycle | Cosine `T_max` shorter than the run (lr cycling) | `T_max=num_epochs`; check the logged lr curve |
| Great accuracy for 5 epochs, then all progress stops | `scheduler.step()` called per batch | Move it to per-epoch; the lr plot would've shown it |
| AMP: "skipping step" spam or loss `nan` early | Scaler calibrating (normal if brief) / genuinely unstable lr | Brief = ignore; persistent = lower lr, check data |
| Shape error at the flatten boundary after porting to CIFAR | `7*7` carried over from Fashion-MNIST | Redo the arithmetic for 32×32 (→ `4*4`), or dummy-tensor trick |
| Residual block: "size mismatch" at the `+` | Shortcut not transformed when channels/stride changed | Add the 1×1-conv shortcut branch |

---

## How Phase 3 sets up Phase 4

This phase looks like "more image classification," but nearly every piece is transformer prep in disguise:

| Phase 3 idea | Where it returns in Phase 4 |
|---|---|
| Residual connections (`out = F(x) + x`) | The transformer block is *built* on them: `x = x + attn(x)`, `x = x + ffwd(x)` |
| Dropout | `dropout = 0.2` sits inside the GPT config — same layer, same train/eval rules |
| AdamW + weight decay | The standard GPT optimizer, verbatim |
| Cosine LR scheduling | The GPT training run uses the same decay idea (plus warmup) |
| AMP (`autocast` + `GradScaler`) | Non-optional at 8 GB: the Phase 4 config is sized assuming it's on |
| Reading train-vs-val curves | How you'll decide when the GPT is overfitting Tiny Shakespeare |
| "Define a block, stack N of them" | `n_layer = 6` — the whole model is one block repeated |
| One-change-at-a-time experiments + named runs | How you'll tune `block_size`/`batch_size`/`n_embd` without fooling yourself |
| BatchNorm's normalize-between-layers idea | Reappears as `nn.LayerNorm` — same purpose, sequence-friendly form |

The deep lesson this time isn't a single architecture trick — it's a **working method**: baseline → diagnose from the curves → change one thing → measure → write it down. Phase 4 hands you a model with a dozen coupled hyperparameters and an 8 GB budget; that method is what makes it navigable.

---

## Suggested next moves

1. Run the **no-augmentation baseline** first and screenshot the diverging curves. Don't skip this — the whole phase is calibrated against that picture.
2. Layer in the tools **one per run** — augmentation, dropout, weight decay, cosine, AMP — each with its own `runs/NN_name` folder and a line in `experiments.md`. Build the ledger from 3.7 with your own numbers.
3. Do the two highest-value break-its if you do nothing else: **remove `model.eval()` with dropout active**, and **ablate the ResNet skip connection**. Both are one-line changes with unforgettable curves.
4. Build the `SmallResNet`, beat 93%, and stare at `out = out + self.shortcut(x)` until it looks obvious. You'll meet that line again, wearing attention, in Phase 4.
5. ✅ commit, then take the temperature of your ambition: if you're itching to start the GPT, you're ready — Phase 3 gave you every tool it needs.
