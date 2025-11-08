import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from fish_speech.models.vqgan.inference import load_model as load_vqgan_model
import torch.nn.functional as F

wav_path = Path("ref.wav")   # your voice sample
out_path = Path("fake.npy")  # where tokens will be saved

vqgan_config = "firefly_gan_vq"
vqgan_checkpoint = Path(r"D:\projects and code\fs2\Voice-cloning\batch_fish_speech\checkpoints\fish-speech-1.5\firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
device = "cpu"   # use "cuda:0" if you have CUDA PyTorch installed

print("Loading VQGAN...")
vqgan = load_vqgan_model(vqgan_config, vqgan_checkpoint, device=device)

print("Loading WAV...")
wav, sr = sf.read(str(wav_path))
if wav.ndim > 1:
    wav = wav[:, 0]  # mono

# Resample to VQGAN's expected sample rate if needed (no extra deps)
target_sr = vqgan.spec_transform.sample_rate
if sr != target_sr:
    x = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,T]
    new_len = int(round(wav.shape[0] * target_sr / sr))
    x = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
    wav = x.squeeze(0).squeeze(0).numpy()
    sr = target_sr

print("Encoding reference voice...")
x = torch.tensor(wav, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,T]
audio_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)

with torch.no_grad():
    indices, _ = vqgan.encode(x, audio_lengths)  # <- pass audio_lengths
    indices = indices[0].detach().cpu().numpy().astype(np.int32)  # [n_codebooks, T_codes]

np.save(out_path, indices)
print(f"✅ Saved voice tokens to {out_path}")
