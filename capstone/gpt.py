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
