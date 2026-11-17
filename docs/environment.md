# Environment

Recorded 2026-11-10 (Phase 1). Re-check after any driver or `pixi.lock` change.

## Hardware
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU, compute capability 12.0 (Blackwell, sm_120) |
| VRAM | 8151 MiB reported by `nvidia-smi`; 7.96 GiB total per `torch.cuda.mem_get_info` (6.8 GiB free at idle) |
| Driver | 610.60 (KMD), CUDA UMD 13.3 |
| CPU / RAM | Intel Core 7 240H, 16 logical cores; 15 GiB RAM visible to WSL |
| Disk | 741 GB free on the WSL root filesystem at start |
| OS | Ubuntu 26.04 LTS on WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2) |

## Software (locked in `pixi.lock`)
| Package | Version |
|---|---|
| pixi | 0.81.0 |
| Python | 3.12.16 |
| PyTorch | 2.14.1, conda-forge `pytorch-gpu`, CUDA 12.9 build, cuDNN 9.10.2; `sm_120` in arch list |
| transformers | 5.22.2 |
| datasets | 5.1.0 |
| transformer-lens | 4.3.0 (PyPI) |
| safetensors | 0.8.0 |
| numpy / pandas | 2.5.4 / 3.1.1 |

CUDA PyTorch follows the pixi docs (pixi.prefix.dev/latest/python/pytorch): conda-forge
`pytorch-gpu` with `cuda = "12.8"` declared on the platform, and `cuda-version >= 12.8`
pinned because Blackwell GPUs need CUDA 12.8 or newer. `torch.cuda.is_available()` is True
and a GPU matmul runs.

## Base model (measured, not assumed)
| Item | Value |
|---|---|
| Checkpoint | `roneneldan/TinyStories-8M` at HF commit `8612e3b15c66ffa94eaa6ee0de5c96edd2d630af` |
| Architecture | GPT-Neo: 8 layers, d_model 256, 16 heads, alternating global/local attention (window 256), GELU MLP |
| Parameters | **19,702,528 total**; 13,390,080 in embeddings (token 50257x256 + position 2048x256); 6,312,448 non-embedding. The output layer shares weights with the token embedding. |
| Checkpoint on disk | 112.4 MB (`pytorch_model.bin`, fp32). Expect ~39 MB per model as fp16 safetensors. |
| Tokenizer | GPT-2 BPE, vocab 50257; `<|endoftext|>` (id 50256) is BOS, EOS and the story separator; no pad token |

## Dataset
`roneneldan/TinyStories` at HF commit `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`:
2,119,719 train stories (471,872,517 tokens), 21,990 validation stories.
Tokenized corpus cached in `data_cache/` (uint16, 926 MB, gitignored).
The validation split is **not** in random order: stories 0-199 have clean loss 1.13 vs ~1.55-1.58 for later ranges, so evaluation sets are always random samples from an index pool.

## TransformerLens
TransformerLens 4 removed `HookedTransformer`. The replacement is
`TransformerBridge.boot_transformers(name)`, which wraps the HF model with **raw, unprocessed
weights**. Layer-norm folding and centring happen only if `enable_compatibility_mode()` is
called, which this project never does. Check (run `20261110-093624-check_tl`): HF clean loss
1.12955847, bridge 1.12955844 on 200 validation stories; difference 2.6e-8 nats/token;
max |logit difference| 3.2e-5. All training and saved weights use the HF model.

## Caches
HF cache (`~/.cache/huggingface`, model + dataset): 3.0 GB, outside the project disk total.
