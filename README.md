
# ✅ Fish-Speech 1.5 Inference (Windows-Friendly, Python-Only)

This repository contains **fully working Python scripts** for running **Fish-Speech 1.5 voice cloning** without needing the long and error-prone command-line process shown in the official repo.

The official Fish-Speech inference workflow requires multiple terminal commands, manual token extraction, and often fails on Windows due to:

- CUDA / Triton / bf16 incompatibility (especially GPUs like GTX 1650)
- Folder path issues (`config.json` not found, tokenizer not found)
- dtype and device mismatches (`cpu vs cuda`, `int64 vs int32`)
- VQGAN prompt extraction not documented properly
- Errors when using Windows paths with spaces

✅ This repo fixes all of that by converting the entire process into **simple Python scripts** you can just run.

---

## 🚀 What this repo does

| Feature | Official Repo | This Repo |
|---------|---------------|-----------|
| Needs terminal commands | ✅ Yes | ❌ No |
| Requires manual prompt extraction | ✅ Yes | ❌ Auto-generated `fake.npy` |
| Crashes on Windows paths | ✅ Common | ❌ Fixed |
| Fails on GTX 1650 (no bfloat16 / Triton) | ✅ Yes | ❌ Disabled Triton + fp16 only |
| One-shot text → voice cloning | ❌ Not provided | ✅ Yes |
| Saves multiple outputs (`output1.wav`, `output2.wav`, …) | ❌ No | ✅ Yes |

---

## 📂 Folder Structure

```
fish-speech1.5-inference/
│── fish_inf2.py            # End-to-end cloning (WAV → fake.npy → generated speech)
│── fish_inference.py       # Uses pre-existing fake.npy for fast text cloning
│── extract_prompt.py       # (optional) WAV → fake.npy only
│── clone_and_speak.py      # alternative chaining script
│── input/
│   └── ref.wav             # Your reference voice (3–10 sec clean speech)
│── output/
│   ├── output1.wav
│   ├── output2.wav
│   └── fake.npy
│── checkpoints/
│   └── fish-speech-1.5/    # model files (NOT included in repo)
│── requirements.txt
│── README.md
│── .gitignore
```

---

## 🔧 Installation

1️⃣ Clone the repo  
```bash
git clone https://github.com/yourname/fish-speech1.5-inference.git
cd fish-speech1.5-inference
```

2️⃣ Create & activate virtual environment  
```bash
python -m venv venv
venv\Scripts\activate   # (Windows)
```

3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

4️⃣ Download Fish-Speech 1.5 checkpoints manually and place under:

```
checkpoints/fish-speech-1.5/
    config.json
    model files...
    firefly-gan-vq-fsq-8x1024-21hz-generator.pth
```

🛑 (They are **NOT** included here due to size + licensing.)

---

## 🎤 How to Run Inference

### ✅ 1. Full automatic (WAV → fake.npy → speech outputs)

Place your reference voice at:

```
input/ref.wav
```

Then run:

```bash
python fish_inf2.py
```

You will get:

```
output/output1.wav
output/output2.wav
...
```

### ✅ 2. If you already have `fake.npy`

```bash
python fish_inference.py
```

### ✅ 3. Extract speaker prompt only (optional)

```bash
python extract_prompt.py
```

It will generate:

```
output/fake.npy
```

---

## ⚙️ Key Fixes Made vs. Official Repo

| Problem in Official Code | Fix in This Repo |
|--------------------------|------------------|
| Model expects folder, not config path | ✅ `load_model(ckpt_dir_folder)` fixed |
| VQGAN encode missing `audio_lengths` arg | ✅ Added correctly |
| Prompt tokens had wrong dtype (`int64`) | ✅ Converted to `int32` before saving |
| "device mismatch: cpu vs cuda" errors | ✅ Fixed by keeping prompt on same device |
| Triton compile crash on GTX GPUs | ✅ Disabled `compile=True`, used fp16 |
| Windows path with spaces breaks Torch | ✅ Used `raw strings r""` everywhere |
| Output overwrote same file | ✅ Now auto-names `output1.wav`, `output2.wav` |

---

## 🧠 Why this repo exists

The Fish-Speech team only provides CLI-based inference scripts that **break on Windows**, require CLI knowledge, and don’t support automated multi-output generation.  
So this repo:

✅ Converts all inference steps into clean Python scripts  
✅ Works on Windows + low-VRAM GPUs (GTX 1650, 4–6GB)  
✅ Removes need to run 5 different commands manually  
✅ Lets you use **your own WAV** and get cloned speech in 1 step  

---

## 📜 Credits

- Original model: **Fish-Speech** (MIT License)
- Scripts rewritten + fixed by **Shreyas Patil**

---

## ⭐ If this repo helped you

Please star the repo and share it — the official repo offers no beginner-friendly inference for Windows users.
