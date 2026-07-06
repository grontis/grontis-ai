# Phase 2 Deep Dive — Your First Real Model: an Image Classifier

Companion notes for **Phase 2** of `pytorch-char-gpt-guide.md` (sections 2.1–2.4).
Detailed explanations + runnable examples so the whole classification pipeline — data → model → loss → training → evaluation → engineering — is second nature before you touch a transformer.

This picks up exactly where the Phase 1 deep dive left off. You already own the five-move training skeleton (`forward → loss → zero_grad → backward → step`). Phase 2 is that skeleton wearing its first real costume: a model with thousands of parameters, a real dataset that arrives in batches, and a loss function built for classification instead of regression.

> **How to read this.** Type every snippet yourself, run it, then break it. The break-it experiments at the end of each section are not optional garnish — they're where the mental model forms. All the code assumes you're inside your `.venv` from Phase 0.

---

## The mental model for the whole phase

Before any code, hold this picture in your head:

```
28×28 grayscale image  ─►  MODEL  ─►  10 numbers (one per class)  ─►  argmax  ─►  a guess
   (the input)                          ("logits")                            ("Sneaker")
```

Everything in Phase 2 is about making that middle box better. The MLP is a first attempt. The CNN is a smarter attempt that *respects the fact the input is a 2D picture*. The engineering layer (2.4) is what turns "I ran a script once" into "I can reproduce, measure, and save this."

The single most important number is **test accuracy**: the fraction of images the model classifies correctly on data it *never trained on*. Regression in Phase 1 asked "how close is my number?" Classification asks "did I pick the right bucket?" — a different question that needs different loss and different metrics.

---

# 2.1 Data — Fashion-MNIST and the `DataLoader`

## What Fashion-MNIST actually is

Fashion-MNIST is 70,000 grayscale images of clothing, each **28×28 pixels**, sorted into **10 classes** (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). It's split into **60,000 training** images and **10,000 test** images. It's a drop-in replacement for the classic handwritten-digit MNIST — same shape, same size — but harder, so an accuracy number actually reflects whether your model is any good. On MNIST almost anything hits 99%; on Fashion-MNIST you have to earn it.

Each pixel is an intensity from 0 (black) to 255 (white). Each label is an integer `0–9`.

## Getting the data with `torchvision`

`torchvision.datasets.FashionMNIST` downloads and caches it for you:

```python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor

train_data = datasets.FashionMNIST(
    root="data",          # where to store the files on disk
    train=True,           # the 60k training split
    download=True,        # fetch it if not already present
    transform=ToTensor(), # how to convert each raw image (see below)
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,          # the 10k test split
    download=True,
    transform=ToTensor(),
)

print(len(train_data), len(test_data))   # 60000 10000

img, label = train_data[0]               # indexing gives ONE (image, label) pair
print(img.shape, img.dtype)              # torch.Size([1, 28, 28]) torch.float32
print(label)                             # an int 0-9
```

Two things worth stopping on:

- **`root="data"`** matches the `data/` line in your `.gitignore` from Phase 0. Datasets don't belong in git — they're big and reproducible from code.
- **A `Dataset` is just an indexable collection.** `train_data[i]` returns one `(image, label)` tuple. It knows its length (`len`) and how to fetch item `i`. That's the entire `Dataset` contract. The `DataLoader` (next) is what turns this into training-ready batches.

## `ToTensor()` — the transform doing quiet, critical work

`transform=ToTensor()` is not decoration. It does two things every time an image is fetched:

1. **Converts** the raw image (a PIL image / NumPy array of `uint8` in `[0, 255]`) into a `float32` tensor.
2. **Scales** pixel values from `[0, 255]` down to `[0.0, 1.0]` by dividing by 255.

It also arranges the tensor as **`(channels, height, width)`** — here `(1, 28, 28)`, the `1` being the single grayscale channel. PyTorch's conv layers expect channels-first, so this ordering matters later.

Why the `[0,1]` scaling matters: neural nets train far better when inputs are small and centered. Feeding raw values up to 255 makes gradients huge and unstable — a fast track to the `nan` loss in the debugging table. Verify it yourself:

```python
img, _ = train_data[0]
print(img.min().item(), img.max().item())   # ~0.0 ... ~1.0, never 255
```

### Normalization: the common next step

`ToTensor()` gets you to `[0,1]`. Many recipes go one further and **normalize** to roughly zero-mean, unit-variance, because optimizers like inputs centered on 0:

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),                     # -> [0,1], shape (1,28,28)
    transforms.Normalize(mean=(0.2860,), std=(0.3530,)),  # -> ~zero-mean
])
```

`Compose` chains transforms left-to-right. The two numbers are Fashion-MNIST's known pixel mean and std (single-element tuples because there's one channel). `Normalize` computes `(x - mean) / std` per channel. For Phase 2 you can start with plain `ToTensor()` and add `Normalize` as a break-it experiment — you'll typically see a small accuracy bump and faster convergence. (Data *augmentation* transforms like random crops/flips belong to Phase 3; here we keep the input deterministic.)

## The `DataLoader` — batching, shuffling, parallel loading

A `Dataset` gives you items one at a time. Training wants **batches**. The `DataLoader` wraps a dataset and hands you batches on demand:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# Peek at one batch
images, labels = next(iter(train_loader))
print(images.shape)   # torch.Size([64, 1, 28, 28])  <- 64 images stacked
print(labels.shape)   # torch.Size([64])             <- 64 integer labels
```

The `DataLoader` gives you three things (the guide's "batches, shuffles, and optionally parallelizes"):

| Feature | What it does | Why it matters |
|---|---|---|
| **Batching** | Stacks `batch_size` items into one tensor with a leading batch dim | GPUs are parallel machines; processing 64 at once is far faster than 64 separate passes, and gradients averaged over a batch are less noisy |
| **Shuffling** (`shuffle=True`) | Reorders the data each epoch | Stops the model from learning the *order* of examples; gives different batch compositions each epoch |
| **Parallel loading** (`num_workers=N`) | Uses N background processes to prep the next batch while the GPU is busy | Keeps the GPU fed so it isn't waiting on the CPU |

Key conventions:

- **`shuffle=True` for training, `shuffle=False` for test/eval.** During training, shuffling helps. During evaluation you're just measuring, so order is irrelevant — leave it off for reproducible, comparable runs.
- **The leading dimension is always the batch.** Every tensor flowing through your model has shape `(batch, ...)`. `images` above is `(64, 1, 28, 28)`: 64 images, 1 channel, 28×28. Internalize this — 90% of shape bugs are about losing track of the batch dim.
- **The last batch may be smaller.** 60,000 isn't divisible by 64, so the final batch of an epoch has 60000 % 64 = 32 images. That's fine; your code should never hardcode 64. (Pass `drop_last=True` if you need every batch identical in size — rarely needed in Phase 2.)

### A note on `num_workers` on Windows

On Windows, `num_workers > 0` spawns subprocesses that **re-import your script**, so your training code *must* be guarded by `if __name__ == "__main__":` or you'll get a spawning error / infinite recursion. Start with `num_workers=0` (simplest, always works) and only add workers once everything runs:

```python
if __name__ == "__main__":
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    # ... rest of training ...
```

For Fashion-MNIST on an RTX 3070 Ti, the dataset is tiny and `num_workers=0` is perfectly fine. This becomes a real lever with bigger datasets in later phases.

### Break-it experiments — 2.1

- **Print shapes at every step.** `next(iter(train_loader))` and inspect `.shape`, `.dtype`, `.min()`, `.max()`. Confirm the batch dim, the `[0,1]` range, and `float32`.
- **Set `batch_size=1`** and re-peek — shape becomes `(1, 1, 28, 28)`. Then `batch_size=60000` — one giant batch (watch memory). Feel how batch size trades update frequency vs. smoothness.
- **Visualize a batch** to prove the labels line up:

  ```python
  import matplotlib.pyplot as plt
  classes = train_data.classes   # ['T-shirt/top', 'Trouser', ...]
  images, labels = next(iter(train_loader))
  fig, axes = plt.subplots(1, 5, figsize=(10, 2))
  for i, ax in enumerate(axes):
      ax.imshow(images[i].squeeze(), cmap="gray")  # squeeze drops the channel dim
      ax.set_title(classes[labels[i]])
      ax.axis("off")
  plt.show()
  ```

---

# 2.2 Start with an MLP (Multi-Layer Perceptron)

## The idea: flatten the image and run it through dense layers

An MLP is the simplest neural network: a stack of `nn.Linear` layers (each fully connected to the next) with a nonlinearity between them. To feed a 2D image in, we **flatten** the `28×28` grid into a single 784-element vector (`28 × 28 = 784`). This throws away the 2D spatial structure — the MLP has no idea that two pixels are neighbors. That's exactly the weakness the CNN in 2.3 fixes, and seeing the accuracy gap is the whole point.

```
image (1,28,28) ─► Flatten ─► (784) ─► Linear ─► ReLU ─► Linear ─► ReLU ─► Linear ─► (10)
                                        784→256          256→128          128→10
```

## Building it with `nn.Module`

Here's the MLP as a proper `nn.Module` (the Phase 1.4 pattern — layers in `__init__`, computation in `forward`):

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()                 # ALWAYS call this first
        self.flatten = nn.Flatten()        # (B,1,28,28) -> (B,784)
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),       # 784 -> 256
            nn.ReLU(),
            nn.Linear(256, 128),           # 256 -> 128
            nn.ReLU(),
            nn.Linear(128, 10),            # 128 -> 10  (one logit per class)
        )

    def forward(self, x):
        x = self.flatten(x)                # collapse the image to a vector
        logits = self.net(x)               # raw scores, NOT probabilities
        return logits
```

Line-by-line intuition:

- **`nn.Flatten()`** turns `(B, 1, 28, 28)` into `(B, 784)`. It leaves the batch dimension alone and flattens the rest — this is why the batch dim discipline from 2.1 matters.
- **`nn.Linear(in, out)`** is `y = xW^T + b`. `Linear(784, 256)` holds a weight matrix of shape `(256, 784)` and a bias of `(256,)` — that's `784*256 + 256 = 200,960` learnable numbers in this one layer alone. This is where the model's capacity lives.
- **`nn.ReLU()`** is the nonlinearity: `relu(x) = max(0, x)`. Without a nonlinearity between the linears, stacking them would collapse into a single linear layer (matrix times matrix is just another matrix) — the model could only ever learn straight-line relationships. ReLU is what lets the network bend, so it can represent complicated decision boundaries.
- **`nn.Sequential`** just chains modules so calling it runs them in order. Purely convenience.
- The final layer outputs **10 raw numbers called logits** — one per class. They are *not* probabilities (they can be negative, and don't sum to 1). Converting them to probabilities is the loss function's job, next.

### Why no softmax at the end?

You might expect a `softmax` to turn the 10 logits into probabilities. **Don't add one here.** `nn.CrossEntropyLoss` (below) applies softmax internally, in a numerically stable way. Adding your own softmax before it applies it *twice* — a subtle, silent bug that hurts training. Output raw logits; let the loss handle the rest.

## The loss: `nn.CrossEntropyLoss`

Regression used MSE ("how far off is my number?"). Classification uses **cross-entropy** ("how much probability did I put on the *correct* class?"). It rewards being confidently right and heavily punishes being confidently wrong.

```python
loss_fn = nn.CrossEntropyLoss()

# preds: (B, 10) raw logits from the model
# targets: (B,) integer class labels 0-9  — NOT one-hot!
loss = loss_fn(preds, targets)
```

Two things that trip up beginners:

1. **It takes raw logits, not probabilities.** Internally it does `log_softmax` then negative-log-likelihood, fused for numerical stability. That's why the model ends at a bare `Linear`.
2. **Targets are plain integer labels**, shape `(B,)`, dtype `long`. You do *not* one-hot encode them yourself — `CrossEntropyLoss` handles that. Passing one-hot vectors, or float labels, is a common shape/dtype error.

Sanity check the scale: with 10 classes, a *random* untrained model should give a loss around `ln(10) ≈ 2.30` (it's spreading probability evenly over 10 options). If your very first loss is near 2.3, your pipeline is wired correctly. If it's wildly larger, something's off (often a double-softmax or unscaled inputs).

## The optimizer: Adam

```python
import torch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

This one line wires up the thing that actually *changes your model's weights*. `backward()` only **measures** — it fills in each weight's gradient ("which way, and how much, would nudging this weight change the loss?"). The **optimizer** is the object that reads those gradients and performs the update when you call `optimizer.step()`. It's the "learning" half of the machine. Take the line apart piece by piece:

- **`torch.optim.Adam`** — the *algorithm*. In Phase 1 you used plain SGD, the simplest possible rule: `w ← w - lr * grad` (every weight, same fixed step). **Adam** (short for *Adaptive Moment Estimation*) is smarter in two ways: it keeps a running average of recent gradients (**momentum** — smooths out mini-batch noise, rolls through flat spots), and it tracks each gradient's squared magnitude to give **every parameter its own effective step size** (weights with consistently large gradients take smaller, more cautious steps). The payoff: it converges faster and is far more forgiving of the learning-rate value than SGD — which is why `Adam(..., lr=1e-3)` is the near-universal default first thing to reach for.

- **`model.parameters()`** — *what* the optimizer is allowed to update. This returns an iterator over every learnable tensor in the model — all the weight matrices and bias vectors of your three `nn.Linear` layers. `nn.Module` tracks them for you; you never list them by hand. Crucially, the optimizer holds a **reference** to these exact tensors, not a copy. That's the invisible link that makes the loop work: `backward()` writes into each tensor's `.grad` field, and because the optimizer points at those same tensors, `step()` can read `.grad` and update the tensor in place. Whatever you pass here is exactly what gets trained — hand it a subset and the rest stays frozen (the mechanism behind fine-tuning).

- **`lr=1e-3`** — the **learning rate** (`0.001`), the single most important knob. It scales how big a step `step()` takes; Adam then adapts *around* it per parameter. Too **high** → steps overshoot, loss bounces or diverges to `nan` (the "crank lr to 1.0" break-it experiment). Too **low** → learns correctly but painfully slowly. `1e-3` is the standard well-behaved default *for Adam specifically* (SGD usually wants something larger like `0.01`–`0.1`). Start here; tune it only once everything else works.

The line itself looks passive because its real job happens later, inside `train_one_epoch`:

```python
optimizer.zero_grad()   # reset last step's gradients to zero
loss.backward()         # autograd fills each param's .grad
optimizer.step()        # optimizer reads .grad, updates each param
```

So the constructor is really just **binding three things together**: the algorithm (`Adam`), the exact tensors it may change (`model.parameters()`), and the step scale (`lr`). After that, the model, the loss, and the optimizer are all talking about the same underlying weight tensors.

One habit-forming note for later phases: this *interface is identical* no matter which optimizer you pick. Swap `Adam` for `SGD` or `AdamW` and the rest of your loop — `zero_grad`, `step` — doesn't change a character. That's the "the skeleton is model-agnostic" lesson from Phase 1, applied to optimizers.

## The two functions: `train_one_epoch` and `evaluate`

The guide asks for two functions built on the 1.3 skeleton. Here they are, fully worked. Study the differences between them — that contrast is a big chunk of the lesson.

```python
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()                         # training mode (matters once you add dropout)
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)             # 1. forward -> (B,10) logits
        loss = loss_fn(preds, labels)     # 2. measure

        optimizer.zero_grad()             # 3. clear old grads
        loss.backward()                   # 4. compute grads
        optimizer.step()                  # 5. update weights

        # ---- bookkeeping for reporting (not part of learning) ----
        running_loss += loss.item() * images.size(0)   # sum, weighted by batch size
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total       # avg loss, accuracy


def evaluate(model, loader, loss_fn, device):
    model.eval()                          # eval mode: dropout off, batchnorm frozen
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():                 # no gradients: faster, less memory
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)

            running_loss += loss.item() * images.size(0)
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total
```

### The four differences between train and eval — memorize these

| | `train_one_epoch` | `evaluate` |
|---|---|---|
| Mode | `model.train()` | `model.eval()` |
| Gradients | tracked (needed for backward) | `with torch.no_grad()` — off |
| Weight updates | `zero_grad → backward → step` | **none** — we only look |
| Data | training set, shuffled | test/val set, not shuffled |

- **`model.eval()` vs `model.train()`** flip the behavior of layers like Dropout and BatchNorm. Your MLP has neither yet, so it makes no difference *now* — but building the habit here means you won't get bitten by the "eval accuracy weirdly low" bug when you add Dropout in Phase 3. Set them always.
- **`torch.no_grad()`** during eval: you're not learning, so recording the autograd graph is pure wasted time and memory.

### How accuracy is computed — unpack this line

```python
correct += (preds.argmax(1) == labels).sum().item()
```

Walk it right to left:

- `preds` is `(B, 10)` logits. `preds.argmax(1)` takes the index of the largest logit **along dimension 1** (the class dimension), giving `(B,)` predicted class indices. The largest logit = the model's pick. (We don't need softmax for this — softmax is monotonic, so the argmax of logits equals the argmax of probabilities.)
- `== labels` compares element-wise against the true labels → a `(B,)` boolean tensor (`True` where correct).
- `.sum()` counts the `True`s (booleans sum as 1/0). `.item()` pulls the Python number out of the one-element tensor.

Dividing the running total by `total` at the end gives the epoch's accuracy as a fraction (0.88 = 88%).

> **Why `loss.item() * images.size(0)` and then divide by `total`?** `loss` is already the *mean* over the batch. Multiplying by the batch size recovers the *sum*, so that after summing across all batches and dividing by the total number of examples, you get a correct dataset-wide average — even when the last batch is smaller than the rest. Naively averaging the per-batch means would slightly over-weight that small last batch.

## Putting it together — a complete, runnable MLP script

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# --- data ---
train_data = datasets.FashionMNIST("data", train=True,  download=True, transform=ToTensor())
test_data  = datasets.FashionMNIST("data", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=64, shuffle=False)

# --- model / loss / optimizer ---
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256), nn.ReLU(),
            nn.Linear(256, 128),     nn.ReLU(),
            nn.Linear(128, 10),
        )
    def forward(self, x):
        return self.net(self.flatten(x))

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- train_one_epoch / evaluate (same as defined above) ---
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        preds = model(images)             # 1. forward -> (B,10) logits
        loss = loss_fn(preds, labels)     # 2. measure

        optimizer.zero_grad()             # 3. clear old grads
        loss.backward()                   # 4. compute grads
        optimizer.step()                  # 5. update weights

        running_loss += loss.item() * images.size(0)
        correct += (preds.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
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

# --- the loop ---
for epoch in range(5):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"epoch {epoch+1} | train acc {tr_acc:.3f} | test acc {te_acc:.3f}")
```

**Target: ~88% test accuracy in a handful of epochs.** On your GPU this trains in seconds. If you land around 0.87–0.89, you've done it right.

### Break-it experiments — 2.2

- **Remove the `nn.ReLU()` lines.** Predict first, then run. Accuracy craters toward a linear model's ceiling — proof that the nonlinearity is what gives the network its power.
- **Add your own `nn.Softmax(dim=1)` before returning logits.** Watch training get worse (double softmax). Then remove it and recover. This burns the "logits, not probabilities" rule into memory.
- **Crank `lr` to `1.0`.** Loss goes to `nan` — the appendix's "LR too high" row, live.
- **Print the first batch's loss before any training.** Confirm it's near `ln(10) ≈ 2.30`.
- **Make it wider/deeper** (e.g., 512 then 256 units). Note accuracy barely moves — the MLP has hit the ceiling of what "ignore the 2D structure" can do. That ceiling is your motivation for 2.3.

---

# 2.3 Upgrade to a CNN and feel the difference

## Why convolutions beat flattening

The MLP flattened the image and lost the fact that pixels have neighbors. A **convolutional layer** keeps the 2D grid and slides a small learnable filter (a "kernel," e.g. 3×3) across it, computing a weighted sum at each position. This bakes in two properties that match how images actually work:

- **Locality:** a feature (an edge, a texture) is defined by nearby pixels, and the kernel only ever looks at a small neighborhood.
- **Translation invariance / weight sharing:** the *same* kernel is reused at every position. A vertical-edge detector works whether the edge is top-left or bottom-right. This also means far fewer parameters than a dense layer, applied everywhere.

Early conv layers learn simple features (edges, gradients); stacked deeper, they compose into complex ones (textures → sleeves → whole garments). This is why the CNN jumps to ~90–92% with the *same training code*.

## The building blocks

```
image (B,1,28,28)
  ─► Conv2d(1→32, 3x3, pad=1) ─► ReLU ─► MaxPool2d(2)   # -> (B,32,14,14)
  ─► Conv2d(32→64, 3x3, pad=1) ─► ReLU ─► MaxPool2d(2)   # -> (B,64,7,7)
  ─► Flatten ─► Linear(64*7*7 → 128) ─► ReLU ─► Linear(128 → 10)
```

- **`nn.Conv2d(in_channels, out_channels, kernel_size, padding)`**
  - `in_channels`: channels coming in (1 for grayscale; 32 for the second conv because the first produced 32).
  - `out_channels`: how many different filters to learn — each produces one output "feature map." More filters = more kinds of features detected = more capacity.
  - `kernel_size=3`: a 3×3 window.
  - `padding=1`: adds a 1-pixel border of zeros so a 3×3 kernel leaves the height/width unchanged (28→28). Without padding, each conv shrinks the image at the edges.
- **`nn.ReLU()`**: same nonlinearity as before, applied per pixel of each feature map.
- **`nn.MaxPool2d(2)`**: slides a 2×2 window and keeps the max in each, halving height and width (28→14→7). This shrinks the spatial size (less compute, bigger effective field of view) and adds a little robustness to small shifts.

### Spatial-size arithmetic — the part people fumble

You **must** be able to compute the shape at each stage, because the first `Linear` after flattening needs the exact number of features. For a conv with `padding=1, stride=1, kernel=3`, size is preserved. Each `MaxPool2d(2)` halves it:

```
28  ── conv(pad=1) ──► 28  ── maxpool(2) ──► 14
14  ── conv(pad=1) ──► 14  ── maxpool(2) ──►  7
```

After the second block: **64 channels × 7 × 7 = 3136 features.** So the first linear is `nn.Linear(64*7*7, 128)`. Get this number wrong and you get a shape-mismatch error at the flatten boundary — the single most common CNN bug. The general convolution output formula, worth memorizing:

```
out = floor((in + 2*padding - kernel) / stride) + 1
```

## Building the CNN

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,1,28,28) -> (B,32,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (B,64,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # -> (B,64,7,7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # -> (B, 64*7*7=3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),                           # -> (B,10) logits
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

**The payoff line from the guide:** you swap `model = MLP()` for `model = CNN()` and *change nothing else*. Same `train_one_epoch`, same `evaluate`, same loss, same optimizer, same loop. That's the deep lesson of Phase 1 paying off — the training skeleton is model-agnostic. Run it and watch test accuracy climb from ~88% to ~90–92%.

> **This contrast is the entire point of 2.3.** You didn't tune hyperparameters or train longer. You changed the *architecture* to one that respects the structure of the data, and you proved it improved things with a number. That "architecture matters, and I can measure it" instinct is what you carry into every later phase.

### A tip: let the model tell you the flatten size

Hand-computing `64*7*7` is error-prone as nets get deeper. A robust trick — run a dummy tensor through the feature extractor once and read the shape:

```python
with torch.no_grad():
    dummy = torch.zeros(1, 1, 28, 28)
    n_features = CNN().features(dummy).flatten(1).shape[1]
print(n_features)   # 3136
```

Or sidestep it entirely with `nn.LazyLinear(128)`, which infers its input size on the first forward pass. Either way, understand the arithmetic first — the shortcut is only safe once you know what it's shortcutting.

### Break-it experiments — 2.3

- **Delete `padding=1`** from both convs. The 28→26→13→11→5 shrinkage changes the flatten size; you'll hit a shape error. Fix it by recomputing — this drills the arithmetic.
- **Remove the `MaxPool2d` layers.** The flatten size explodes to `64*28*28`, the linear layer balloons, memory/compute jump, and accuracy may not improve — pooling earns its place.
- **Compare parameter counts.** Print `sum(p.numel() for p in model.parameters())` for the MLP vs the CNN. The CNN often has *fewer* parameters yet does better — weight sharing at work.
- **Bump `out_channels`** (32→64, 64→128) and see the accuracy/compute trade-off.

---

# 2.4 Add the engineering layer — make it a real project

A model that trains once in a script you can't reproduce isn't a project. These four habits are the "software development in the AI space" skill the guide keeps emphasizing. Fold each into your script.

## Reproducibility — `torch.manual_seed`

Neural nets are full of randomness: weight initialization, data shuffling, dropout. Set a seed at the top so runs are repeatable and your experiments are comparable:

```python
torch.manual_seed(42)
```

Re-run twice and confirm you get *identical* numbers. This is what lets you say "change X improved accuracy by 0.4%" and trust it wasn't just luck. For fuller determinism (GPU ops, cuDNN), you'd add:

```python
torch.manual_seed(42)
# torch.cuda.manual_seed_all(42)          # if using CUDA
# torch.backends.cudnn.deterministic = True   # slower, but bit-for-bit repeatable
```

The plain `manual_seed(42)` is enough for Phase 2. Don't expect *bit-identical* results across different machines or PyTorch versions — expect them within the same setup.

## Device handling — define it once, use it everywhere

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN().to(device)                       # move the model's parameters to GPU
# inside the loop, every batch too:
images, labels = images.to(device), labels.to(device)
```

The rule that prevents the "RuntimeError: ... on different devices" bug: **the model and every input tensor must be on the same device.** Move the model once (right after building it) and move each batch inside the loop. If a tensor is still on the CPU while the model is on the GPU, you'll get that error — and the fix is always "find the tensor that missed its `.to(device)`."

## Checkpointing — save your best model

You want to save the model so you can reload it later without retraining. The idiomatic way is to save the **`state_dict`** (a plain dict of all parameter tensors), not the whole model object:

```python
import os
os.makedirs("checkpoints", exist_ok=True)

# --- save (do this when you beat your best test accuracy) ---
torch.save(model.state_dict(), "checkpoints/best.pt")

# --- load later (into a freshly constructed model of the SAME class) ---
model = CNN().to(device)
model.load_state_dict(torch.load("checkpoints/best.pt", map_location=device))
model.eval()   # switch to eval mode before using it for inference
```

Why `state_dict` and not `torch.save(model, ...)`? Saving the whole object pickles the class definition and file paths, which breaks if you refactor. Saving just the weights is portable: reconstruct the architecture in code, then pour the saved weights in. (`checkpoints/` and `*.pt` are already in your `.gitignore` — weights don't go in git.)

The **"save the best" pattern** in a training loop — track the best test accuracy seen and only overwrite when you beat it:

```python
best_acc = 0.0
for epoch in range(num_epochs):
    train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    _, test_acc = evaluate(model, test_loader, loss_fn, device)
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "checkpoints/best.pt")
        print(f"  new best: {best_acc:.3f} — saved")
```

This way `best.pt` always holds your best-ever weights, even if later epochs overfit and get worse. **Practice loading it back** in a separate small script — being able to reload and run a saved model is a core skill you'll lean on constantly.

## TensorBoard — watch training live

Printing numbers works, but *seeing curves* is how practitioners actually work — trends, plateaus, and overfitting jump out visually. `SummaryWriter` logs scalars to disk; TensorBoard renders them in a browser.

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/fashion_cnn")   # 'runs/' is git-ignored

for epoch in range(num_epochs):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    te_loss, te_acc = evaluate(model, test_loader, loss_fn, device)

    writer.add_scalar("loss/train", tr_loss, epoch)
    writer.add_scalar("loss/test",  te_loss, epoch)
    writer.add_scalar("acc/train",  tr_acc,  epoch)
    writer.add_scalar("acc/test",   te_acc,  epoch)

writer.close()
```

Then, in a **second terminal** (venv activated), launch the viewer and open the URL it prints (usually `http://localhost:6006`):

```bash
tensorboard --logdir runs
```

- **`add_scalar(tag, value, step)`**: the tag's `group/name` form (`loss/train`) makes TensorBoard group related curves. `step` is the x-axis (here the epoch).
- **The reason this matters for Phase 3:** when you plot train vs. test accuracy on the same chart, **overfitting becomes visible** — train accuracy keeps climbing while test accuracy stalls or drops, the two curves visibly diverging. You'll learn to read that shape and respond to it (dropout, augmentation, weight decay). Building the logging habit now means the diagnostic is ready when you need it.
- Give each run a distinct subfolder (`runs/fashion_cnn`, `runs/fashion_mlp`) so you can overlay and compare MLP vs. CNN on the same axes — a direct, visual version of the 2.2-vs-2.3 lesson.

### Break-it experiments — 2.4

- **Run with the same seed twice**, confirm identical output. Then *remove* the seed and run twice — outputs now differ. Feel what reproducibility buys you.
- **Comment out one batch's `.to(device)`** and read the exact error message. Now you'll recognize it instantly in the wild.
- **Log both MLP and CNN runs** to `runs/fashion_mlp` and `runs/fashion_cnn`, then view them overlaid in TensorBoard. Your architecture lesson, drawn as two curves.
- **Overtrain on purpose** — run 40+ epochs and watch (in TensorBoard) train accuracy pull away from test accuracy. That gap *is* overfitting; it's the cliffhanger into Phase 3.

---

## How Phase 2 sets up everything after it

Everything here reappears, dressed differently:

| Phase 2 idea | Where it returns |
|---|---|
| `DataLoader`, batches, `train`/`eval` split | Every phase. Phase 4's `get_batch()` is a hand-rolled version of the same idea for text. |
| `CrossEntropyLoss` on logits | Phase 4's GPT predicts the next character with **exactly this loss** over a vocab-sized set of classes. |
| `model.train()` / `model.eval()` + `no_grad()` | Non-negotiable from here on; essential once Dropout/LayerNorm arrive. |
| The model-agnostic training skeleton | Unchanged through the GPT. You'll swap in a transformer and reuse the loop verbatim. |
| Checkpointing, seeds, TensorBoard, device handling | The engineering spine of every later project; Phase 5 formalizes it into clean modules. |
| Watching for overfitting in the curves | The central drama of Phase 3 (regularization, augmentation, AMP). |

The jump from MLP to CNN taught you the meta-lesson that carries all the way to the capstone: **a better architecture, measured honestly, is how progress happens.** The transformer in Phase 4 is just the next, much larger, instance of that same move.

---

## Suggested next moves

1. Build the MLP end-to-end, hit ~88%, then swap in the CNN and confirm the jump to ~90–92% with *no other changes*.
2. Do the break-it experiments — especially "remove ReLU," "double softmax," and "delete padding." Predict each outcome before running.
3. Wire in all four engineering pieces (seed, device, checkpoint-the-best, TensorBoard) and overlay the MLP vs. CNN curves.
4. Deliberately overfit and *see* the train/test gap in TensorBoard — then walk into Phase 3 already knowing what problem it solves.
```