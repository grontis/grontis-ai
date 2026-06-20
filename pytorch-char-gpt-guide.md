# Building a Character-Level GPT From Scratch in PyTorch
### A hands-on learning path, from `torch.tensor` to a working language model

---

## How to use this guide

This is a **ramp**, not a recipe. Each phase builds the muscle you need for the next, and the capstone (a small GPT) only makes sense once the earlier layers are second nature. Resist the urge to skip ahead to Phase 4 — the people who do end up copy-pasting a transformer they can't debug.

A few principles to hold onto:

- **Type the code yourself.** Do not copy-paste the snippets. Re-typing is slow on purpose; it's where the learning happens.
- **Break things deliberately.** After each working example, change something (a shape, a learning rate, a layer size) and predict what will happen *before* you run it. Being wrong is the point.
- **Commit at every milestone.** You'll see `✅ commit` markers throughout. Git is part of the skill set you said you wanted, so we treat it as non-optional from day one.
- **One concept at a time.** If you're confused, you've usually skipped a fundamental. Go back one step.

Rough pacing for someone brand new, working a few evenings a week: Phase 0–1 in week 1, Phase 2 in week 2, Phase 3 in week 3, Phase 4 across weeks 4–6. Slower is completely fine.

---

## Your hardware at a glance

| Component | Spec | What it means for you |
|---|---|---|
| GPU | RTX 3070 Ti Laptop, **8 GB VRAM** | Comfortable for everything in this guide. The 8 GB ceiling is the number you'll plan around in Phase 4. |
| System RAM | 32 GB | Plenty for data loading and preprocessing. |
| Chassis | Alienware x17 R2 (thin 17") | Will thermal-throttle on long runs. Elevate it, maximize airflow, and don't panic when clocks drop. |

**Two habits specific to this machine:**
1. Keep an eye on temps and VRAM with `nvidia-smi` (run `nvidia-smi -l 2` in a terminal to refresh every 2s).
2. When you hit a memory error in Phase 4, the fix is almost always *smaller batch size* or *mixed precision* — both covered below.

---

## Phase 0 — Environment & tooling

Goal: a clean, reproducible project where your GPU is verified working. This is the "software development in the AI space" part you asked for, and it matters more than people admit — most beginner pain is environment pain, not modeling pain.

### 0.1 Install the pieces

1. **Python 3.11 or 3.12** from python.org. (3.10–3.14 all work with current PyTorch; 3.11/3.12 are the safe middle.) During install, check **"Add Python to PATH."**
2. **VS Code** + the official **Python** and **Jupyter** extensions.
3. **Git** from git-scm.com. Configure your name/email:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
4. **Latest NVIDIA Game Ready or Studio driver.** You do **not** need to install the CUDA Toolkit separately — the PyTorch GPU wheel ships with its own CUDA runtime. A current driver is all that's required.

> **Windows-native vs WSL2:** As a beginner on a single GPU, stay on **native Windows** for now. It's simpler and works fine. WSL2 (a Linux environment inside Windows) is worth exploring later, but don't add that variable while you're learning the basics.

### 0.2 Create the project

Open a terminal in a folder where you keep code:

```bash
mkdir char-gpt
cd char-gpt
git init
python -m venv .venv
.venv\Scripts\activate        # Windows. You should see (.venv) in your prompt.
```

> The virtual environment (`.venv`) keeps this project's packages isolated from the rest of your system. Activate it **every time** you open a new terminal for this project. Forgetting to is the #1 beginner confusion ("it says torch isn't installed!" — because you're outside the venv).

### 0.3 Install PyTorch with GPU support

Go to **https://pytorch.org/get-started/locally/** and use the selector: **Stable → Windows → Pip → Python → CUDA (pick the latest offered)**. Copy the command it gives you and run it. It looks like this (your CUDA tag may differ — use the one from the site):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Then install the few extras we'll use:

```bash
pip install numpy matplotlib tensorboard tqdm
```

### 0.4 Verify the GPU — the most important 6 lines you'll write today

Create `check_gpu.py`:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Run it: `python check_gpu.py`. You want `CUDA available: True` and your 3070 Ti's name. **If it says `False`**, your driver or the wrong (CPU-only) wheel is the cause — reinstall using the CUDA index URL above. Don't proceed until this passes.

### 0.5 Lock in good hygiene

Create a `.gitignore`:

```
.venv/
__pycache__/
*.pyc
data/
runs/
checkpoints/
*.pt
```

Freeze your dependencies so the project is reproducible:

```bash
pip freeze > requirements.txt
```

✅ **commit:** `git add . && git commit -m "Phase 0: environment, GPU verified"`

---

## Phase 1 — PyTorch fundamentals (no models yet)

Goal: genuine fluency with tensors, autograd, and the training-loop skeleton. Don't rush this — every later phase is just these ideas wearing a costume. Work in a plain `fundamentals.py`.

> **Keeping the interactive feel in `.py` files.** The one thing notebooks are good for is running a few lines and immediately inspecting a tensor. You get the same thing without leaving `.py` files using **VS Code interactive cells**: put `# %%` on its own line to mark a cell, then press **Shift+Enter** to run just that block in an attached Python session — output, plots, and live variables appear in a side panel you can keep poking at. It's the notebook experience inside a real script. When you want to inspect mid-execution instead, drop `breakpoint()` on any line and run with the debugger. (This is why we installed the Jupyter extension in Phase 0 even though we're not using `.ipynb` files.)

### 1.1 Tensors

A tensor is an n-dimensional array that can live on the GPU and remember how it was computed. Work through, by hand:

- **Creation:** `torch.zeros`, `torch.ones`, `torch.randn`, `torch.arange`, `torch.tensor([...])`.
- **Attributes:** `.shape`, `.dtype`, `.device`. Print these constantly — 90% of bugs are shape/dtype/device mismatches.
- **Indexing & slicing:** like NumPy. `x[0]`, `x[:, 1]`, `x[x > 0]`.
- **Reshaping:** `.view()`, `.reshape()`, `.unsqueeze()`, `.squeeze()`, `.transpose()`. Understand the difference between `view` (needs contiguous memory) and `reshape` (copies if needed).
- **Broadcasting:** how a `(3,1)` tensor and a `(1,4)` tensor combine into `(3,4)`. This trips everyone up; spend real time here.
- **Moving to GPU:** `x = x.to("cuda")`. Operations between tensors on different devices error out — you'll meet this often.

**Exercise:** Without loops, create a 5×5 multiplication table using broadcasting. (Hint: `torch.arange(1,6)` reshaped two ways.)

### 1.2 Autograd — the engine that makes learning possible

This is the conceptual heart of PyTorch. Build the intuition with a tiny example you fully understand:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2          # y = x²
y.backward()        # compute dy/dx
print(x.grad)       # should be 6.0, because dy/dx = 2x = 2*3
```

Internalize the loop: a tensor with `requires_grad=True` records operations into a graph; `.backward()` walks that graph backward and fills in `.grad`; the optimizer later nudges parameters in the direction `.grad` points.

**Exercise (do not skip):** Implement linear regression *by hand* — no `nn.Module`, no optimizer. Generate noisy points around `y = 2x + 1`. Create `w` and `b` with `requires_grad=True`. Loop: compute predictions, compute mean-squared-error loss, call `loss.backward()`, then manually update `w` and `b` with `torch.no_grad(): w -= lr * w.grad` (and zero the grads after). Watch `w` crawl toward 2 and `b` toward 1. **When this clicks, you understand training.** Everything else is scale.

### 1.3 The training-loop skeleton

Every model you ever train has this shape. Memorize it:

```python
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        preds = model(batch_x)          # 1. forward
        loss = loss_fn(preds, batch_y)  # 2. measure error

        optimizer.zero_grad()           # 3. clear old gradients
        loss.backward()                 # 4. compute new gradients
        optimizer.step()                # 5. update weights
```

Steps 3–5 in that order, every time. Forgetting `zero_grad()` is a classic bug that makes gradients pile up and training go haywire.

### 1.4 The `nn.Module` pattern

`nn.Module` is how you package a model: define layers in `__init__`, define the forward pass in `forward`. Read the [PyTorch quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) and rebuild its model from memory.

✅ **commit:** `git commit -am "Phase 1: tensors, autograd, manual training loop"`

---

## Phase 2 — Your first real model: an image classifier

Goal: train a working model end-to-end, then upgrade it and *measure* the improvement. We use Fashion-MNIST (clothing images) — same shape as MNIST but harder, so your accuracy numbers actually mean something.

### 2.1 Data

`torchvision.datasets.FashionMNIST` downloads it for you. Wrap it in `DataLoader`s with `batch_size=64`, `shuffle=True` for training. Understand what a DataLoader does: it batches, shuffles, and (optionally) parallelizes loading.

### 2.2 Start with an MLP

Flatten each 28×28 image to a 784-vector and feed it through a couple of `nn.Linear` layers with `nn.ReLU` between them, ending in 10 outputs (one per class). Loss: `nn.CrossEntropyLoss`. Optimizer: `torch.optim.Adam(model.parameters(), lr=1e-3)`.

Write **two** functions — `train_one_epoch(...)` and `evaluate(...)` — using the skeleton from 1.3. In `evaluate`, wrap everything in `torch.no_grad()` and call `model.eval()` (and `model.train()` before training). Track loss and accuracy each epoch.

**Target:** ~88% test accuracy in a handful of epochs. This trains in seconds on your GPU.

### 2.3 Upgrade to a CNN and feel the difference

Now build a small convolutional net: two `nn.Conv2d` + `nn.ReLU` + `nn.MaxPool2d` blocks, then flatten into linear layers. Convolutions exploit the 2D structure that the MLP threw away. Same training code — just swap the model. You should jump to ~90–92%.

**This contrast is the lesson:** architecture matters, and you just proved it with a number.

### 2.4 Add the engineering layer

Now make it a *real* project, not a throwaway script:

- **Reproducibility:** at the top, set `torch.manual_seed(42)`. Re-run and confirm you get identical results.
- **Device handling:** `device = "cuda" if torch.cuda.is_available() else "cpu"` once at the top, used everywhere.
- **Checkpointing:** save your best model with `torch.save(model.state_dict(), "checkpoints/best.pt")` and practice loading it back.
- **TensorBoard:** log loss/accuracy with `torch.utils.tensorboard.SummaryWriter`, then run `tensorboard --logdir runs` and open the browser. Watching live curves is how real practitioners work.

✅ **commit:** `git commit -am "Phase 2: MLP + CNN classifier with logging and checkpoints"`

---

## Phase 3 — Going deeper with CIFAR-10

Goal: meet the problems that show up at real scale — overfitting, regularization, and fitting work into 8 GB. CIFAR-10 is 32×32 **color** images across 10 classes; meaningfully harder than Fashion-MNIST.

Build a deeper CNN, then layer in these ideas one at a time so you can see what each does:

- **Data augmentation** (`transforms.RandomCrop`, `RandomHorizontalFlip`): cheaply expands your data and fights overfitting.
- **Regularization:** `nn.Dropout` and the `weight_decay` argument in your optimizer.
- **Reading the curves:** when training accuracy climbs but validation stalls or drops, that's overfitting. You'll *see* it in TensorBoard and learn to respond.
- **Learning-rate scheduling:** `torch.optim.lr_scheduler.CosineAnnealingLR` to decay the LR over training.
- **Mixed precision (AMP):** this is your first real VRAM tool. It runs parts of the model in 16-bit, cutting memory use and speeding things up:

  ```python
  scaler = torch.amp.GradScaler("cuda")
  # inside the loop:
  with torch.amp.autocast("cuda"):
      preds = model(batch_x)
      loss = loss_fn(preds, batch_y)
  optimizer.zero_grad()
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  ```

**Stretch goal:** build a small **ResNet** by implementing a residual block (the `out = layers(x) + x` skip connection) and stacking a few. Understanding residual connections now pays off directly in Phase 4 — transformers use the same trick.

✅ **commit:** `git commit -am "Phase 3: CIFAR-10 CNN with augmentation, AMP, scheduling"`

---

## Phase 4 — The capstone: a character-level GPT

Goal: build a transformer language model from scratch that learns to generate text one character at a time. You'll implement self-attention by hand — the single idea behind every modern LLM.

> **Strongly recommended companion:** Andrej Karpathy's free video *"Let's build GPT: from scratch, in code, spelled out"* and his **nanoGPT** repo. This phase deliberately follows the same conceptual arc. Watch a section, then build it yourself in your own file before looking at his code. Use his repo to check yourself, not to copy.

We build it in **six incremental steps**, each runnable. Never write the whole thing at once.

### Step 1 — Data and the language-modeling setup
- Get a plain-text corpus. **Tiny Shakespeare** (~1 MB, a single `.txt`) is the standard starting point; any text you like works.
- Build the **vocabulary**: the sorted set of unique characters. Create `stoi` (char→int) and `itos` (int→char) dictionaries, and `encode`/`decode` functions.
- Encode the whole corpus into one long tensor of integers. Split 90/10 into train/val.
- Write `get_batch()`: pick random starting points, and for each, grab a chunk of length `block_size` as input `x` and the same chunk shifted by one as target `y`. **The task is literally "predict the next character."** Print a batch and stare at it until the shift makes sense.

### Step 2 — A bigram baseline (no attention yet)
Build the dumbest possible language model: a single `nn.Embedding(vocab_size, vocab_size)` where each character directly predicts logits for the next one. Train it with `CrossEntropyLoss`. Write a `generate()` method that samples one character at a time and feeds it back in.

The output will be near-gibberish — **that's correct and important.** You now have the full LM pipeline (data → model → loss → generation) working. Everything from here just makes the model smarter.

### Step 3 — A single self-attention head
This is the centerpiece. Implement one head:
- Project each token into **query**, **key**, and **value** vectors via three `nn.Linear` layers.
- Compute attention scores as `q @ k.transpose(-2, -1)`, scaled by `1/sqrt(head_size)`.
- Apply a **causal mask** (`torch.tril`) so a position can only attend to itself and earlier positions — the model must never peek at the future it's trying to predict.
- `softmax` the scores, then use them to take a weighted sum of the **values**.

Spend time here. Print the attention matrix for a tiny input and confirm it's lower-triangular. If you understand why the mask is triangular, you understand causal attention.

### Step 4 — Multi-head attention + feed-forward
- **Multi-head:** run several heads in parallel and concatenate their outputs. Different heads learn different relationships.
- **Feed-forward:** a small two-layer MLP applied to each position independently, with a ReLU/GELU in the middle. This is where per-token "thinking" happens.

### Step 5 — Assemble the Transformer block, then stack it
A block = multi-head attention + feed-forward, each wrapped with:
- a **residual connection** (`x = x + sublayer(x)` — the skip connection from Phase 3), and
- **`nn.LayerNorm`** applied before each sublayer.

Then build the full model: a token embedding **plus a positional embedding** (so the model knows *where* each character sits), a stack of N blocks, a final LayerNorm, and a linear head projecting to `vocab_size`. Reuse your Step-2 training loop and `generate()` unchanged.

### Step 6 — Train, generate, and tune
A starting configuration that fits comfortably in 8 GB:

```
block_size  = 256     # context length
batch_size  = 32      # drop to 16 if you hit out-of-memory
n_embd      = 384     # embedding dimension
n_head      = 6
n_layer     = 6
dropout     = 0.2
learning_rate = 3e-4
```

Train with AMP (from Phase 3) to save memory and time, and use `nvidia-smi -l 2` in a second terminal to watch VRAM. **If you get a CUDA out-of-memory error:** lower `batch_size` first, then `block_size`, then `n_embd`. Add **temperature** to your sampler (divide logits by a value like 0.8 before softmax) to control how "creative" generation is.

After enough training, your model will produce text that *looks* like Shakespeare — fake archaic words, plausible structure, character names. It won't make sense, and **that's the expected, wonderful result** for a small char-level model. You built the thing that makes ChatGPT-shaped systems work, just tiny.

✅ **commit:** `git commit -am "Phase 4: working character-level GPT"`

---

## Phase 5 — Engineering polish & where to go next

You now have a working model. Turn the project into something you'd be proud to show:

- **Refactor** your scripts into clean modules: `data.py`, `model.py`, `train.py`, `generate.py`. By Phase 4 you'll likely have one large `gpt.py` doing everything — splitting it into focused files (and importing across them) is itself a core software-engineering skill.
- **Config management:** move hyperparameters into a `@dataclass` or a YAML file + `argparse`, so you can run experiments from the command line without editing code.
- **A `README.md`:** what it is, how to set up, how to train, sample output. This is your portfolio piece.
- **One small test:** e.g., assert that `decode(encode(text)) == text`. Your first taste of testing ML code.

**Then pick a direction to grow:**
- Swap the character tokenizer for a **subword (BPE) tokenizer** and watch quality jump.
- Train on **your own** text corpus (your notes, a book you like, code).
- Scale up `n_layer`/`n_embd` until you hit your VRAM wall — and learn **gradient accumulation** to train "bigger" batches than fit at once.
- Learn **fine-tuning**: take a small pretrained model from Hugging Face and adapt it, which is how most real-world AI work actually happens.

---

## Appendix — debugging & common pitfalls

| Symptom | Usual cause | Fix |
|---|---|---|
| `CUDA available: False` | CPU-only wheel installed | Reinstall torch using the CUDA `--index-url` |
| `RuntimeError: ... on different devices` | Some tensor still on CPU | `.to(device)` your model **and** every batch |
| Loss is `nan` | Learning rate too high; bad data | Lower LR; check for `inf`/`nan` in inputs |
| `CUDA out of memory` | Batch/model too big for 8 GB | Smaller `batch_size`, then AMP, then smaller model |
| Loss won't go down | Forgot `optimizer.zero_grad()`; LR too low; bug in shapes | Print shapes; verify the training-loop order |
| Shapes don't match | Reshape/broadcast confusion | Print `.shape` at every step until you find it |
| `eval` accuracy weirdly low | Forgot `model.eval()` / `torch.no_grad()` | Toggle train/eval modes correctly |

**Golden debugging rule:** when stuck, print `.shape`, `.dtype`, and `.device` of the tensors involved. The bug is almost always one of those three.

### Core resources
- **PyTorch official tutorials** — pytorch.org/tutorials (start with "Learn the Basics")
- **PyTorch docs** — pytorch.org/docs (reference, not reading material)
- **Karpathy, "Let's build GPT"** + **nanoGPT** — the definitive Phase 4 companion
- **Karpathy, "Neural Networks: Zero to Hero"** — the whole series is gold for fundamentals

---

*Work through it in order, commit often, and let yourself be slow on Phase 1. The fundamentals you build there are what make the GPT at the end feel inevitable instead of magic.*
