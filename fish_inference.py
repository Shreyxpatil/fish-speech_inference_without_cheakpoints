from fish_speech.models.text2semantic.inference import generate_long, load_model
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
import torch
import numpy as np
import os
import re
import time
from loguru import logger
from pathlib import Path
import soundfile as sf
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# =========================
# User inputs / paths
# =========================
# 1) Reference WAV (3–10s clean speech) — read from input/ref.wav
ref_wav = Path(r"D:\projects and code\fs2\Voice-cloning\batch_fish_speech\input\ref.wav")

# 2) What to say (each item -> one output WAV)
text = [
    "Hello, this is my cloned voice speaking.",
    "This is a second sentence to test voice consistency."
]

# 3) Checkpoints folder (Fish-Speech 1.5)
ckpt_dir = Path(r"D:\projects and code\fs2\Voice-cloning\batch_fish_speech\checkpoints\fish-speech-1.5")

# 4) Firefly-GAN-VQ generator checkpoint (for encode/decode)
vqgan_config = "firefly_gan_vq"
vqgan_checkpoint = ckpt_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"

# 5) Output directory — write to \output and auto-increment filenames
output_dir = Path(r"D:\projects and code\fs2\Voice-cloning\batch_fish_speech\output")

# =========================
# Runtime settings (match your working script)
# =========================
prompt_text = ["Your prompt text"] * len(text)
num_samples = 1
max_new_tokens = 512
chunk_length = 100
top_p = 0.75
repetition_penalty = 1.4
temperature = 0.4
seed = 42
compile = False
device = "cuda:0"
half = False
precision = torch.float16

# =========================
# Helpers
# =========================
def make_fake_npy_from_wav(ref_wav_path: Path, vq_ckpt: Path, save_path: Path):
    """Create speaker tokens (fake.npy) from a reference WAV using Firefly-GAN-VQ."""
    logger.info("Loading VQGAN for prompt-token extraction...")
    vqgan = load_vqgan_model(vqgan_config, str(vq_ckpt), device="cpu")

    logger.info(f"Loading reference WAV: {ref_wav_path}")
    wav, sr = sf.read(str(ref_wav_path))
    if wav.ndim > 1:
        wav = wav[:, 0]  # mono

    target_sr = vqgan.spec_transform.sample_rate
    if sr != target_sr:
        logger.info(f"Resampling from {sr} Hz to {target_sr} Hz for VQGAN.")
        x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,T]
        new_len = int(round(len(wav) * target_sr / sr))
        x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
        wav = x.squeeze(0).squeeze(0).numpy()

    x = torch.tensor(wav, dtype=torch.float32, device="cpu").unsqueeze(0).unsqueeze(0)  # [1,1,T]
    audio_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device="cpu")

    logger.info("Encoding reference voice to prompt tokens (indices)...")
    with torch.no_grad():
        indices, _ = vqgan.encode(x, audio_lengths)               # [1, n_codebooks, T_codes]
        indices = indices[0].detach().cpu().numpy().astype(np.int32)  # keep int32

    np.save(save_path, indices)
    logger.info(f"Saved prompt tokens to {save_path}")
    return save_path

def next_output_index(out_dir: Path) -> int:
    """Returns the current max N for files named outputN.wav (0 if none)."""
    max_n = 0
    for p in out_dir.glob("output*.wav"):
        m = re.fullmatch(r"output(\d+)\.wav", p.name, flags=re.IGNORECASE)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n

# =========================
# Start
# =========================
os.makedirs(output_dir, exist_ok=True)
warnings.filterwarnings('ignore', category=FutureWarning)

# 0) Create fake.npy automatically beside outputs (so it persists)
fake_npy_path = output_dir / "fake.npy"
make_fake_npy_from_wav(ref_wav, vqgan_checkpoint, fake_npy_path)

# 1) Standard variables as in your working script, but now using the generated fake.npy
prompt_tokens = [fake_npy_path] * len(text)

start_time = time.time()
logger.info("Starting single inference script...")

logger.info("Loading model ...")
t0 = time.time()
# pass FOLDER path to load_model
model, decode_one_token = load_model(
    ckpt_dir, device, precision, compile=compile
)
with torch.device(device):
    model.setup_caches(
        max_batch_size=1,
        max_seq_len=model.config.max_seq_len,
        dtype=next(model.parameters()).dtype,
    )
if torch.cuda.is_available():
    torch.cuda.synchronize()
logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

# Load prompt tokens to GPU (keep dtype as int32)
prompt_tokens_ = [torch.from_numpy(np.load(p)).to(device) for p in prompt_tokens]

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# Warmup (same style as your working code)
logger.info("Cold start: compiling model for fast inference...")
t_compile = time.time()
_ = list(generate_long(
    model=model,
    device=device,
    decode_one_token=decode_one_token,
    text="Привет!",
    num_samples=1,
    max_new_tokens=8,
    top_p=top_p,
    repetition_penalty=repetition_penalty,
    temperature=temperature,
    compile=compile,
    chunk_length=chunk_length,
    prompt_text=prompt_text[0] if prompt_text else None,
    prompt_tokens=prompt_tokens_[0] if prompt_tokens_ else None,
))
logger.info(f"Cold start complete. Model is ready for fast inference. Compile time: {time.time() - t_compile:.2f} seconds")

# Load VQGAN once
t_vqgan = time.time()
vqgan_model = load_vqgan_model(vqgan_config, str(vqgan_checkpoint), device=device)
logger.info(f"Loaded VQGAN model in {time.time() - t_vqgan:.2f} seconds")

# Decode helper
def codes_to_wav(codes, output_path, vqgan_model):
    dev = next(vqgan_model.parameters()).device
    codes = torch.cat(codes, dim=1).to(dev)  # keep integer dtype
    if codes.ndim == 3:
        codes = codes[0]
    feature_lengths = torch.tensor([codes.shape[1]], device=dev)
    t0 = time.time()
    fake_audios, _ = vqgan_model.decode(indices=codes[None], feature_lengths=feature_lengths)
    fake_audio = fake_audios[0, 0].float().detach().cpu().numpy()
    t_decode = time.time() - t0
    sf.write(str(output_path), fake_audio, vqgan_model.spec_transform.sample_rate)
    logger.info(f"Decoded and saved audio to {output_path} in {t_decode:.2f} seconds")

# Determine starting index so we keep appending: output1.wav, output2.wav, ...
current_index = next_output_index(output_dir)

# Generate + save (append filenames)
for idx, (t, pt, ptok) in enumerate(zip(text, prompt_text, prompt_tokens_)):
    t_ch = time.time()
    responses = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=t,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=float(top_p),
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        chunk_length=chunk_length,
        prompt_text=pt,
        prompt_tokens=ptok,
    )

    all_codes = []
    for response in responses:
        if hasattr(response, 'action') and response.action == "sample":
            all_codes.append(response.codes)
            logger.info(f"[SINGLE] Generating chunk: idx={idx}, text_len={len(response.text) if response.text else 0}")
        elif hasattr(response, "action") and response.action == "next":
            if all_codes:
                current_index += 1
                wav_path = output_dir / f"output{current_index}.wav"
                codes_to_wav(all_codes, wav_path, vqgan_model)
                logger.info(f"[SINGLE] [{idx}] {t} → {wav_path} (chunks: {len(all_codes)}) | Time for sample: {time.time() - t_ch:.2f} seconds")
                all_codes = []
            logger.info("Finished current sample")

logger.info(f"Total single inference time after compile: {time.time() - start_time:.2f} seconds")
