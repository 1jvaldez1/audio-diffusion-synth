"""Shared module for the diffusion-path comparison (Paths A, B, C, DiT).

Everything that must be identical across paths lives here: config, data
loading, the fold-based split, crop selection, the mel front end, the training
loop, the sampler, and the metrics. Each path notebook supplies only its model,
a `forward_fn`, and a `to_repr` transform.

Usage in a notebook:
    import sys; sys.path.append("/content/drive/MyDrive/thesis")
    import common as C
"""
from __future__ import annotations

import csv
import json
import os
import random
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.training_utils import EMAModel


# ----------------------------------------------------------------------------
# 1. Config
# ----------------------------------------------------------------------------
@dataclass
class Config:
    # audio front end
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64
    spec_frames: int = 128          # time axis of the spectrogram; sets the clip length
    db_range: float = 80.0          # dynamic range kept below the calibrated ceiling

    # data
    num_classes: int = 50
    val_fold: int = 4               # used for model selection; never for the final test
    heldout_fold: int = 5           # ESC-50 fold never seen in training
    crop_energy_frac: float = 0.25  # a crop is valid if its energy >= this fraction of the clip's loudest crop

    # diffusion
    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"
    p_uncond: float = 0.1           # label-dropout rate for classifier-free guidance
    inference_steps: int = 100

    # optimisation
    batch_size: int = 32
    lr: float = 1e-4
    warmup_steps: int = 500
    total_steps: int = 10000
    grad_clip: float = 1.0
    ema_halflife_frac: float = 0.1  # EMA half-life as a fraction of total_steps
    ckpt_every: int = 1000
    milestone_every: int = 5000     # keep a permanent copy of the EMA weights this often
    log_every: int = 100
    seed: int = 42

    # locations
    data_root: str = "./data"
    out_root: str = "/content/drive/MyDrive/thesis_runs"

    @property
    def clip_samples(self) -> int:
        # With center=True the STFT yields 1 + N // hop frames, so this length
        # gives exactly `spec_frames` frames with no padded columns.
        return (self.spec_frames - 1) * self.hop_length

    @property
    def clip_seconds(self) -> float:
        return self.clip_samples / self.sample_rate

    @property
    def null_class(self) -> int:
        return self.num_classes     # extra label index meaning "no label"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 2. Data: ESC-50 loading, fold split, energy-based crops
# ----------------------------------------------------------------------------
ESC50_URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/master"
ESC50_SECONDS = 5


def load_esc50(cfg: Config) -> dict:
    """Return all 2,000 clips as full 5-second waveforms.

    Output dict: waveforms (N, 1, S) float32, labels (N,), folds (N,), class_names.
    Each clip is peak-normalised once as a whole, so crops keep their relative
    level inside the clip. The result is cached on local disk.
    """
    import soundfile as sf

    data_root = Path(cfg.data_root)
    cache = data_root / f"esc50_{cfg.sample_rate}.pt"
    if cache.exists():
        return torch.load(cache)

    root = data_root / "ESC-50-master"
    if not root.exists():
        data_root.mkdir(parents=True, exist_ok=True)
        zip_path = data_root / "esc50.zip"
        print("Downloading ESC-50 (~650 MB)...")
        urllib.request.urlretrieve(ESC50_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_root)
        zip_path.unlink()

    with open(root / "meta" / "esc50.csv") as f:
        rows = list(csv.DictReader(f))

    n_samples = ESC50_SECONDS * cfg.sample_rate
    resamplers: dict[int, torchaudio.transforms.Resample] = {}
    waves = torch.zeros(len(rows), 1, n_samples)
    for i, row in enumerate(rows):
        audio, sr = sf.read(root / "audio" / row["filename"], dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio).mean(dim=1, keepdim=True).T          # (1, samples), mono
        if sr != cfg.sample_rate:
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, cfg.sample_rate)
            wave = resamplers[sr](wave)
        wave = wave[:, :n_samples]
        peak = wave.abs().max()
        if peak > 1e-8:
            wave = wave / peak * 0.95
        waves[i, :, : wave.shape[-1]] = wave

    names = {int(r["target"]): r["category"] for r in rows}
    data = {
        "waveforms": waves,
        "labels": torch.tensor([int(r["target"]) for r in rows]),
        "folds": torch.tensor([int(r["fold"]) for r in rows]),
        "class_names": [names[k] for k in sorted(names)],
    }
    torch.save(data, cache)
    return data


def split_by_fold(data: dict, cfg: Config) -> tuple[dict, dict]:
    """Split into (train, heldout) by ESC-50 fold.

    Fragments of one source recording always share a fold, so splitting by
    fold keeps them on one side. A random clip-level split would leak.
    """
    held = data["folds"] == cfg.heldout_fold
    val = data["folds"] == cfg.val_fold

    def take(mask):
        return {
            "waveforms": data["waveforms"][mask],
            "labels": data["labels"][mask],
            "folds": data["folds"][mask],
            "class_names": data["class_names"],
        }

    return take(~held & ~val), take(val), take(held)


def valid_crop_starts(waveforms: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, torch.Tensor]:
    """Find, for every clip, which crop positions contain the sound.

    Returns starts (n_starts,) in samples and valid (N, n_starts) bool. A start
    is valid when the crop's mean energy is at least `crop_energy_frac` of the
    loudest crop in that clip. The loudest crop is always valid.
    """
    L, hop = cfg.clip_samples, cfg.hop_length
    energies = []
    for i in range(0, len(waveforms), 256):
        sq = waveforms[i : i + 256] ** 2
        energies.append(F.avg_pool1d(sq, kernel_size=L, stride=hop).squeeze(1))
    energy = torch.cat(energies)                                   # (N, n_starts)
    valid = energy >= cfg.crop_energy_frac * energy.max(dim=1, keepdim=True).values
    starts = torch.arange(energy.shape[1]) * hop
    return starts, valid


class CropDataset(torch.utils.data.Dataset):
    """Training set: a fresh random valid crop of each clip on every access.

    Yields (waveform (1, clip_samples), label). The representation (mel,
    waveform, linear STFT) is applied later by each path's `to_repr`, so every
    path trains on the same underlying audio.
    """

    def __init__(self, split: dict, cfg: Config):
        self.wave, self.labels, self.L = split["waveforms"], split["labels"], cfg.clip_samples
        self.starts, valid = valid_crop_starts(self.wave, cfg)
        self.valid_idx = [torch.nonzero(v).squeeze(1) for v in valid]

    def __len__(self) -> int:
        return len(self.wave)

    def __getitem__(self, i: int):
        idx = self.valid_idx[i]
        s = int(self.starts[idx[torch.randint(len(idx), (1,))]])
        return self.wave[i, :, s : s + self.L], self.labels[i]


def reference_crops(split: dict, cfg: Config, k: int = 2, return_clip_index: bool = False):
    """Deterministic crops for evaluation: up to k valid crops per clip, evenly spread.

    Use on the held-out split for the Frechet reference, and on the train split
    for the memorisation check. With return_clip_index=True a third tensor gives,
    for every crop, the index of its clip within `split`, so crops of one clip
    (or one source recording) can be kept together in splits and permutations.
    """
    starts, valid = valid_crop_starts(split["waveforms"], cfg)
    L, crops, labels, clip_idx = cfg.clip_samples, [], [], []
    for i, v in enumerate(valid):
        idx = torch.nonzero(v).squeeze(1)
        pick = idx[torch.linspace(0, len(idx) - 1, k).round().long().unique()]
        for s in starts[pick].tolist():
            crops.append(split["waveforms"][i, :, s : s + L])
            labels.append(split["labels"][i])
            clip_idx.append(i)
    out = (torch.stack(crops), torch.stack(labels))
    return out + (torch.tensor(clip_idx),) if return_clip_index else out


# ----------------------------------------------------------------------------
# 3. Mel front end
# ----------------------------------------------------------------------------
class MelFrontEnd:
    """Waveform -> log-mel in [-1, 1], with one fixed dB window for all data.

    The window is [db_max - db_range, db_max]. `db_max` is calibrated once on
    the training split and saved, so real audio, Path A samples, and Path B's
    generated audio all pass through the identical transform. A fixed window
    (not a per-clip one) keeps absolute level, so loudness errors stay visible.
    """

    def __init__(self, cfg: Config, db_max: float | None = None, device="cpu"):
        self.cfg, self.db_max, self.device = cfg, db_max, device
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, power=2.0,
        ).to(device)

    def raw_db(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, 1, samples) -> unnormalised log-mel power in dB, (B, 1, n_mels, frames)."""
        return 10.0 * torch.log10(self.mel(wave.to(self.device)).clamp(min=1e-10))

    @torch.no_grad()
    def fit(self, waveforms: torch.Tensor, q: float = 0.999, batch: int = 64) -> "MelFrontEnd":
        """Set db_max to the q-quantile of training-set dB values."""
        g = torch.Generator().manual_seed(self.cfg.seed)
        vals = []
        for i in range(0, len(waveforms), batch):
            db = self.raw_db(waveforms[i : i + batch]).flatten().cpu()
            vals.append(db[torch.randint(len(db), (20000,), generator=g)])   # subsample: quantile has a size limit
        self.db_max = float(torch.quantile(torch.cat(vals), q))
        return self

    @torch.no_grad()
    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, 1, clip_samples) -> (B, 1, n_mels, spec_frames) in [-1, 1]."""
        assert self.db_max is not None, "call fit() or load() first"
        db = self.raw_db(wave)
        assert db.shape[-1] == self.cfg.spec_frames, f"got {db.shape[-1]} frames; expected clips of {self.cfg.clip_samples} samples"
        lo = self.db_max - self.cfg.db_range
        return (db.clamp(lo, self.db_max) - lo) / self.cfg.db_range * 2.0 - 1.0

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"db_max": self.db_max, "config": asdict(self.cfg)}, f, indent=2)

    @classmethod
    def load(cls, path, cfg: Config, device="cpu") -> "MelFrontEnd":
        with open(path) as f:
            return cls(cfg, db_max=json.load(f)["db_max"], device=device)


class LinearFrontEnd(MelFrontEnd):
    """Waveform -> log linear-frequency spectrogram in [-1, 1] (Path C).

    Identical to MelFrontEnd except for the filterbank: the STFT is pooled into
    `n_mels` triangular bands of equal width in Hz instead of mel-spaced bands.
    The output has the same shape as the mel spectrogram, so Paths A and C
    differ only in how the frequency axis is laid out. The dB window is
    calibrated separately, because linear bands carry different levels.
    """

    def __init__(self, cfg: Config, db_max: float | None = None, device="cpu"):
        self.cfg, self.db_max, self.device = cfg, db_max, device
        self.spec = torchaudio.transforms.Spectrogram(n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=2.0).to(device)
        self.fb = torchaudio.functional.linear_fbanks(
            n_freqs=cfg.n_fft // 2 + 1, f_min=0.0, f_max=cfg.sample_rate / 2,
            n_filter=cfg.n_mels, sample_rate=cfg.sample_rate,
        ).to(device)                                                    # (n_freqs, n_bands)

    def raw_db(self, wave: torch.Tensor) -> torch.Tensor:
        power = self.spec(wave.to(self.device))                         # (B, 1, n_freqs, frames)
        bands = (power.transpose(-1, -2) @ self.fb).transpose(-1, -2)   # (B, 1, n_bands, frames)
        return 10.0 * torch.log10(bands.clamp(min=1e-10))


@torch.no_grad()
def linear_to_mel(x: torch.Tensor, linear_fe: LinearFrontEnd, mel_fe: MelFrontEnd) -> torch.Tensor:
    """Convert normalised linear-band spectrograms to the mel front end's space.

    Each linear band's power is spread evenly over the STFT bins it covers, and
    the mel filterbank is applied to that estimate. The step is approximate, so
    always pass the real reference through the same conversion before comparing
    Path C against it.
    """
    cfg = linear_fe.cfg
    w_lin = linear_fe.fb.to(x.device)                                   # (n_freqs, n_bands)
    w_mel = mel_fe.mel.mel_scale.fb.to(x.device)                        # (n_freqs, n_mels)
    per_bin = w_lin / w_lin.sum(dim=0, keepdim=True).clamp(min=1e-10)   # band power -> mean power per bin
    per_bin = per_bin / w_lin.sum(dim=1, keepdim=True).clamp(min=1e-10) # average the bands overlapping each bin
    mapping = w_mel.T @ per_bin                                         # (n_mels, n_bands)
    
    lo_lin = linear_fe.db_max - cfg.db_range
    power = 10.0 ** (((x + 1.0) / 2.0 * cfg.db_range + lo_lin) / 10.0)
    mel_db = 10.0 * torch.log10((mapping @ power).clamp(min=1e-10))
    lo_mel = mel_fe.db_max - cfg.db_range
    return (mel_db.clamp(lo_mel, mel_fe.db_max) - lo_mel) / cfg.db_range * 2.0 - 1.0


@torch.no_grad()
def spec_to_wave(x: torch.Tensor, frontend: MelFrontEnd, n_iter: int = 64, batch: int = 64,
                 seed: int = 0) -> torch.Tensor:
    """Normalised band spectrogram (Path A mel or Path C linear) -> waveform via Griffin-Lim.

    Steps: undo the fixed dB window to get band power, spread band power back to
    the 513 STFT bins with the pseudo-inverse of the filterbank (the same
    least-squares estimate torchaudio's InverseMelScale makes), then estimate
    phase with Griffin-Lim. Neither path has a vocoder, so this is how their
    samples become audio. The loss it causes is part of each path's
    representation ceiling: pass real clips through the same function and score
    them like the generated ones.

    x: (B, 1, n_bands, spec_frames) in [-1, 1]. Returns (B, 1, clip_samples) on CPU.
    """
    cfg, dev = frontend.cfg, frontend.device
    fb = frontend.fb if isinstance(frontend, LinearFrontEnd) else frontend.mel.mel_scale.fb   # (n_freqs, n_bands)
    inv = torch.linalg.pinv(fb.T.double()).float().to(dev)                                      # (n_freqs, n_bands)
    gl = torchaudio.transforms.GriffinLim(n_fft=cfg.n_fft, hop_length=cfg.hop_length, power=2.0,
                                          n_iter=n_iter, length=cfg.clip_samples).to(dev)
    lo = frontend.db_max - cfg.db_range
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    for i in range(0, len(x), batch):
        xb = x[i : i + batch, 0].to(dev).float()                                    # (b, n_bands, T)
        power = 10.0 ** (((xb + 1.0) / 2.0 * cfg.db_range + lo) / 10.0)
        stft_power = (inv @ power).clamp(min=0.0)                                   # (b, n_freqs, T)
        torch.manual_seed(int(torch.randint(2**31 - 1, (1,), generator=g)))         # Griffin-Lim's random phase init
        out.append(gl(stft_power).unsqueeze(1).cpu())   # not clipped: Griffin-Lim can overshoot, and clipping would add distortion
    return torch.cat(out)


# ----------------------------------------------------------------------------
# 3b. Codec front end (tokenized Path B)
# ----------------------------------------------------------------------------
class CodecFrontEnd:
    """Waveform <-> DAC tokens, and the continuous token embeddings Path B-codec diffuses over.

    DAC (Descript Audio Codec, 16 kHz model) encodes audio at 50 frames per second.
    Each frame is described by 12 residual codebooks of 1,024 entries; every
    entry is an 8-dimensional vector. A frame's tokens are therefore 12 integers,
    and their embeddings concatenate to a 96-dimensional vector.

    The diffusion model works on those 96-dimensional token embeddings,
    standardised per channel with statistics fitted on the training split. To
    turn a generated sample back into audio, each 8-dimensional chunk is snapped
    to its nearest codebook entry (giving real tokens), and DAC's decoder
    synthesises the waveform from the tokens.

    Clips are zero-padded from `clip_samples` to a whole number of frames that
    divides by 8 (104 frames for the default config), so the 1D UNet can halve
    the sequence three times. Decoded audio is cropped back to `clip_samples`.
    """

    MODEL_ID = "descript/dac_16khz"

    def __init__(self, cfg: Config, n_codebooks: int = 12, mean=None, std=None, device="cpu",
                 model=None, frame_multiple: int = 8):
        from transformers import DacModel

        self.cfg, self.device, self.n_codebooks = cfg, device, n_codebooks
        self.model = (model if model is not None else DacModel.from_pretrained(self.MODEL_ID)).to(device).eval()
        self.model.requires_grad_(False)
        assert self.model.config.sampling_rate == cfg.sample_rate, "DAC model and Config disagree on sample rate"
        self.hop = int(np.prod(self.model.config.downsampling_ratios))
        frames = -(-cfg.clip_samples // self.hop)
        self.frames = -(-frames // frame_multiple) * frame_multiple
        self.padded_samples = self.frames * self.hop
        q = self.model.quantizer.quantizers[:n_codebooks]
        self.code_dim = q[0].codebook.weight.shape[1]
        self.channels = n_codebooks * self.code_dim
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32).view(1, -1, 1).to(device)
        self.std = None if std is None else torch.as_tensor(std, dtype=torch.float32).view(1, -1, 1).to(device)

    # -- tokens ---------------------------------------------------------------
    def _pad(self, wave: torch.Tensor) -> torch.Tensor:
        wave = wave.to(self.device).float()
        return F.pad(wave, (0, self.padded_samples - wave.shape[-1]))

    @torch.no_grad()
    def tokens(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, 1, clip_samples) -> integer tokens (B, n_codebooks, frames)."""
        return self.model.encode(self._pad(wave), n_quantizers=self.n_codebooks).audio_codes[:, : self.n_codebooks]

    @torch.no_grad()
    def embed_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer tokens -> unstandardised token embeddings (B, channels, frames)."""
        q = self.model.quantizer.quantizers
        return torch.cat([q[i].codebook(codes[:, i]).transpose(1, 2) for i in range(self.n_codebooks)], dim=1)

    @torch.no_grad()
    def snap(self, emb: torch.Tensor) -> torch.Tensor:
        """Unstandardised embeddings -> nearest tokens (B, n_codebooks, frames).

        Euclidean distance to the raw codebook vectors, because the model is
        trained to reproduce those raw vectors. (DAC's own encoder uses cosine
        distance on normalised vectors; that is for its encoder output, not for
        reconstructing a codebook entry.)
        """
        q, d, codes = self.model.quantizer.quantizers, self.code_dim, []
        for i in range(self.n_codebooks):
            z = emb[:, i * d : (i + 1) * d].transpose(1, 2)                       # (B, frames, d)
            book = q[i].codebook.weight                                           # (K, d)
            dist = (z ** 2).sum(-1, keepdim=True) - 2 * z @ book.T + (book ** 2).sum(-1)
            codes.append(dist.argmin(-1))
        return torch.stack(codes, dim=1)

    @torch.no_grad()
    def tokens_to_wave(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer tokens -> (B, 1, clip_samples) on CPU."""
        wave = self.model.decode(audio_codes=codes.to(self.device)).audio_values
        return wave[..., : self.cfg.clip_samples].unsqueeze(1).clamp(-1, 1).cpu()

    @torch.no_grad()
    def roundtrip(self, wave: torch.Tensor, batch: int = 32) -> torch.Tensor:
        """Real audio -> tokens -> audio. Path B-codec's representation ceiling."""
        return torch.cat([self.tokens_to_wave(self.tokens(wave[i : i + batch])) for i in range(0, len(wave), batch)])

    # -- what the diffusion model sees ------------------------------------------
    @torch.no_grad()
    def fit(self, waveforms: torch.Tensor, batch: int = 32) -> "CodecFrontEnd":
        """Per-channel mean and std of the token embeddings over the given crops."""
        s = s2 = n = 0
        for i in range(0, len(waveforms), batch):
            e = self.embed_tokens(self.tokens(waveforms[i : i + batch])).double()
            s = s + e.sum(dim=(0, 2)); s2 = s2 + (e ** 2).sum(dim=(0, 2)); n += e.shape[0] * e.shape[2]
        mean = s / n
        std = (s2 / n - mean ** 2).clamp(min=1e-8).sqrt()
        self.mean = mean.float().view(1, -1, 1).to(self.device)
        self.std = std.float().view(1, -1, 1).to(self.device)
        return self

    @torch.no_grad()
    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, 1, clip_samples) -> standardised token embeddings (B, channels, frames)."""
        assert self.mean is not None, "call fit() or load() first"
        return (self.embed_tokens(self.tokens(wave)) - self.mean) / self.std

    @torch.no_grad()
    def from_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer tokens (B, n_codebooks, frames) -> standardised token embeddings.

        The training transform when tokens are cached (TokenCropDataset): no
        encoder pass, only a table lookup.
        """
        return (self.embed_tokens(codes.to(self.device)) - self.mean) / self.std

    @torch.no_grad()
    def encode_clips(self, waveforms: torch.Tensor, batch: int = 16) -> torch.Tensor:
        """Encode whole clips once: (N, 1, samples) -> int16 tokens (N, n_codebooks, frames) on CPU.

        Clips are zero-padded to a whole number of frames. Each clip is encoded
        in one pass, so every token sees the clip's surrounding audio.
        """
        n = -(-waveforms.shape[-1] // self.hop) * self.hop
        out = []
        for i in range(0, len(waveforms), batch):
            w = F.pad(waveforms[i : i + batch].to(self.device).float(), (0, n - waveforms.shape[-1]))
            out.append(self.model.encode(w, n_quantizers=self.n_codebooks).audio_codes[:, : self.n_codebooks].short().cpu())
        return torch.cat(out)

    @torch.no_grad()
    def to_wave(self, x: torch.Tensor, batch: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated standardised embeddings -> (waveform (B, 1, clip_samples), tokens). CPU tensors."""
        waves, codes = [], []
        for i in range(0, len(x), batch):
            c = self.snap(x[i : i + batch].to(self.device) * self.std + self.mean)
            waves.append(self.tokens_to_wave(c)); codes.append(c.cpu())
        return torch.cat(waves), torch.cat(codes)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"model_id": self.MODEL_ID, "n_codebooks": self.n_codebooks,
                       "mean": self.mean.flatten().tolist(), "std": self.std.flatten().tolist(),
                       "config": asdict(self.cfg)}, f)

    @classmethod
    def load(cls, path, cfg: Config, device="cpu", model=None) -> "CodecFrontEnd":
        with open(path) as f:
            d = json.load(f)
        return cls(cfg, n_codebooks=d["n_codebooks"], mean=d["mean"], std=d["std"], device=device, model=model)


class LatentFrontEnd:
    """Waveform <-> DAC's continuous latent, reduced with PCA (Path B-latent).

    DAC's decoder does not read tokens directly. It reads a continuous 1,024-dim
    vector per 20 ms frame: the sum of the 12 codebook vectors, each projected
    back to 1,024 dims. That latent is smooth (nearby latents decode to similar
    audio), which is what Gaussian diffusion needs and what the stacked raw
    codebook vectors of CodecFrontEnd lacked.

    1,024 dims per frame is too many to diffuse efficiently, and most of them
    carry little variance. PCA, fitted on training-set frames only, keeps the
    top `k` directions. Decoding inverts the PCA (dropped directions are set to
    their training mean) and runs DAC's decoder on the continuous latent:
    there is no snapping step.

    Scaling: by default the k components are divided by ONE global scale so
    that their average variance is 1. That keeps the PCA's ordering of
    importance in the loss: high-variance directions (which carry most of the
    audio) cost more to get wrong than low-variance ones. whiten=True instead
    scales every component to unit variance, giving every direction equal
    weight; that is the equal-weighting that let the fine codebooks swamp the
    loss in the token-embedding run, so it is kept only as an ablation.

    Tokens remain available: `to_wave` also re-quantizes the generated latent
    with DAC's own quantizer and returns the 12 codes per frame.
    """

    def __init__(self, codec: CodecFrontEnd, k: int = 64, whiten: bool = False,
                 mean=None, components=None, eigvals=None):
        self.codec, self.cfg, self.device = codec, codec.cfg, codec.device
        self.whiten = whiten
        self.mean = None if mean is None else torch.as_tensor(mean, dtype=torch.float32, device=self.device)
        self.components = None if components is None else torch.as_tensor(components, dtype=torch.float32, device=self.device)
        self.eigvals = None if eigvals is None else torch.as_tensor(eigvals, dtype=torch.float32, device=self.device)
        if self.components is not None:
            self.set_k(k)
        else:
            self.k = k

    # -- the 1,024-dim latent ----------------------------------------------------
    @torch.no_grad()
    def latent_from_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """Integer tokens (B, n_codebooks, T) -> DAC's quantized latent (B, 1024, T)."""
        return self.codec.model.quantizer.from_codes(codes.to(self.device).long())[0]

    @torch.no_grad()
    def latent(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, 1, clip_samples) -> DAC's quantized latent (B, 1024, frames)."""
        return self.latent_from_tokens(self.codec.tokens(wave))

    # -- PCA -------------------------------------------------------------------------
    @torch.no_grad()
    def fit(self, windows, batch: int = 32) -> "LatentFrontEnd":
        """Fit the PCA on token windows from the training set.

        `windows` is a sequence of integer token arrays (n_codebooks, frames),
        for example a TokenCropDataset sampled a few times per clip, so the PCA
        sees the same distribution of frames the diffusion model trains on.
        The covariance is accumulated in float64, so any number of frames fits.
        """
        s = s2 = None; n = 0
        for i in range(0, len(windows), batch):
            codes = torch.stack([torch.as_tensor(windows[j]) for j in range(i, min(i + batch, len(windows)))])
            z = self.latent_from_tokens(codes).double()                    # (b, D, T)
            f = z.permute(0, 2, 1).reshape(-1, z.shape[1])                 # frames as rows
            s = f.sum(0) if s is None else s + f.sum(0)
            s2 = f.T @ f if s2 is None else s2 + f.T @ f
            n += f.shape[0]
        mean = s / n
        cov = s2 / n - torch.outer(mean, mean)
        vals, vecs = torch.linalg.eigh(cov)                                # ascending
        order = torch.argsort(vals, descending=True)
        self.mean = mean.float()
        self.eigvals = vals[order].clamp(min=0).float()
        self.components = vecs[:, order].T.contiguous().float()           # (D, D), rows = directions
        self.set_k(self.k)
        return self

    def set_k(self, k: int) -> "LatentFrontEnd":
        """Choose how many PCA directions to keep (no refit needed)."""
        self.k = int(k)
        ev = self.eigvals[: self.k]
        if self.whiten:
            self.scale = ev.clamp(min=1e-12).sqrt().view(1, -1, 1)
        else:
            self.scale = ev.mean().clamp(min=1e-12).sqrt().view(1, 1, 1).expand(1, self.k, 1)
        return self

    def explained(self, k: int | None = None) -> float:
        """Fraction of the latent's variance kept by the top k directions."""
        k = self.k if k is None else k
        return float(self.eigvals[:k].sum() / self.eigvals.sum())

    @torch.no_grad()
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 1024, T) latent -> (B, k, T) scaled PCA coefficients."""
        W = self.components[: self.k]                                      # (k, D)
        return torch.einsum("kd,bdt->bkt", W, z - self.mean.view(1, -1, 1)) / self.scale

    @torch.no_grad()
    def unproject(self, x: torch.Tensor) -> torch.Tensor:
        """(B, k, T) scaled coefficients -> (B, 1024, T) latent (dropped directions at their mean)."""
        W = self.components[: self.k]
        return torch.einsum("kd,bkt->bdt", W, x.to(self.device).float() * self.scale) + self.mean.view(1, -1, 1)

    # -- what the diffusion model sees -------------------------------------------------
    @torch.no_grad()
    def from_tokens(self, codes: torch.Tensor) -> torch.Tensor:
        """Training transform with cached tokens: tokens -> latent -> PCA coefficients."""
        return self.project(self.latent_from_tokens(codes))

    @torch.no_grad()
    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """(B, 1, clip_samples) -> (B, k, frames) scaled PCA coefficients."""
        return self.project(self.latent(wave))

    @torch.no_grad()
    def to_wave(self, x: torch.Tensor, batch: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Generated coefficients -> (waveform (B, 1, clip_samples), re-quantized tokens). CPU tensors."""
        waves, codes = [], []
        for i in range(0, len(x), batch):
            z = self.unproject(x[i : i + batch])
            w = self.codec.model.decoder(z)[..., : self.cfg.clip_samples]
            waves.append(w.reshape(len(z), 1, -1).clamp(-1, 1).cpu())
            c = self.codec.model.quantizer(z, self.codec.n_codebooks)[1][:, : self.codec.n_codebooks]
            codes.append(c.cpu())
        return torch.cat(waves), torch.cat(codes)

    @torch.no_grad()
    def roundtrip(self, wave: torch.Tensor, batch: int = 32) -> torch.Tensor:
        """Real audio -> tokens -> latent -> top-k PCA -> latent -> audio. Path B-latent's ceiling."""
        return torch.cat([self.to_wave(self(wave[i : i + batch]))[0] for i in range(0, len(wave), batch)])

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": self.mean.cpu(), "components": self.components.cpu(), "eigvals": self.eigvals.cpu(),
                    "n_codebooks": self.codec.n_codebooks, "model_id": self.codec.MODEL_ID,
                    "config": asdict(self.cfg)}, path)

    @classmethod
    def load(cls, path, codec: CodecFrontEnd, k: int = 64, whiten: bool = False) -> "LatentFrontEnd":
        d = torch.load(path, map_location="cpu")
        assert d["n_codebooks"] == codec.n_codebooks, f"{path} was fitted with {d['n_codebooks']} codebooks"
        return cls(codec, k=k, whiten=whiten, mean=d["mean"], components=d["components"], eigvals=d["eigvals"])


class TokenCropDataset(torch.utils.data.Dataset):
    """Training set for Path B-codec with the tokens computed once, not every step.

    Every training clip is encoded whole by the codec a single time. Each access
    then cuts a fresh random window of `codec.frames` frames out of the stored
    tokens, so the model still sees a new random crop of each clip every epoch,
    but training no longer runs the codec encoder.

    Differences from encoding each crop on the fly (CropDataset + CodecFrontEnd):
      * Crops start on the codec's frame grid (every 320 samples = 20 ms)
        instead of every `hop_length` samples (16 ms).
      * Tokens near a window's edges were computed with the clip's neighbouring
        audio in view, instead of with the crop's zero padding. The last frames
        of a window hold real audio rather than encoded silence; they fall
        beyond `clip_samples` and are cropped off after decoding.
    A window is valid by the same rule as CropDataset: the energy of its first
    `clip_samples` samples is at least `crop_energy_frac` of the clip's loudest.

    Yields (tokens (n_codebooks, frames) int64, label). Use `codec.from_tokens`
    as the training transform.
    """

    def __init__(self, split: dict, codec: CodecFrontEnd, cfg: Config, tokens: torch.Tensor | None = None):
        self.labels, self.frames = split["labels"], codec.frames
        self.tokens = codec.encode_clips(split["waveforms"]) if tokens is None else tokens
        n_frames = self.tokens.shape[-1]
        assert n_frames >= self.frames, "clips are shorter than one training window"
        energies = []
        for i in range(0, len(split["waveforms"]), 256):
            sq = split["waveforms"][i : i + 256] ** 2
            energies.append(F.avg_pool1d(sq, kernel_size=cfg.clip_samples, stride=codec.hop).squeeze(1))
        energy = torch.cat(energies)[:, : n_frames - self.frames + 1]          # one entry per frame-aligned start
        valid = energy >= cfg.crop_energy_frac * energy.max(dim=1, keepdim=True).values
        self.valid_idx = [torch.nonzero(v).squeeze(1) for v in valid]

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, i: int):
        idx = self.valid_idx[i]
        f = int(idx[torch.randint(len(idx), (1,))])
        return self.tokens[i, :, f : f + self.frames].long(), self.labels[i]


def build_unet2d(cfg: Config, in_channels: int = 1, dropout: float = 0.1):
    """The class-conditional 2D UNet shared by Paths A and C (and the DiT baseline)."""
    from diffusers import UNet2DModel

    return UNet2DModel(
        sample_size=(cfg.n_mels, cfg.spec_frames),
        in_channels=in_channels,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256),
        norm_num_groups=32,
        dropout=dropout,
        attention_head_dim=32,
        resnet_time_scale_shift="scale_shift",
        num_class_embeds=cfg.num_classes + 1,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


# ----------------------------------------------------------------------------
# 4. Diffusion: schedulers, training loop, sampler
# ----------------------------------------------------------------------------
def make_schedulers(cfg: Config) -> tuple[DDPMScheduler, DDIMScheduler]:
    """Return (training scheduler, fast sampler) built from the same noise schedule."""
    train = DDPMScheduler(
        num_train_timesteps=cfg.num_train_timesteps,
        beta_schedule=cfg.beta_schedule,
        prediction_type=cfg.prediction_type,
    )
    # "trailing" makes sampling start from the last (noisiest) training timestep.
    sampler = DDIMScheduler.from_config(train.config, timestep_spacing="trailing")
    return train, sampler


def train_diffusion(model, forward_fn, to_repr, dataset, cfg: Config, run_dir, device, conditional: bool = True):
    """Train any path with the same recipe. Resumes from run_dir/last.pt if present.

    model       nn.Module predicting the scheduler's target.
    forward_fn  (model, x_noisy, t, labels) -> prediction, same shape as x_noisy.
                `labels` is None when conditional=False.
    to_repr     (waveform batch on device) -> clean sample x0 in the path's space.
    Returns (ema, losses) where losses is the mean loss per `log_every` window.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "last.pt"
    scheduler, _ = make_schedulers(cfg)
    use_amp = torch.device(device).type == "cuda"

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, cfg.warmup_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_decay = 0.5 ** (1.0 / max(1.0, cfg.total_steps * cfg.ema_halflife_frac))
    ema = EMAModel(model.parameters(), decay=ema_decay)
    step, losses = 0, []

    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        assert ck["config"]["total_steps"] == cfg.total_steps, \
            f"{ckpt_path} was trained with total_steps={ck['config']['total_steps']}; use an empty run_dir"
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); lr_sched.load_state_dict(ck["lr_sched"]); scaler.load_state_dict(ck["scaler"])
        step, losses = ck["step"], ck["losses"]
        print(f"Resumed from step {step}")

    def save():
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "lr_sched": lr_sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step, "losses": losses, "config": asdict(cfg)}, tmp)
        os.replace(tmp, ckpt_path)          # atomic: a disconnect mid-save cannot corrupt the checkpoint

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    def batches():
        while True:
            yield from loader

    model.train()
    window = []
    for wave, labels in batches():
        if step >= cfg.total_steps:
            break
        x0 = to_repr(wave.to(device))
        noise = torch.randn_like(x0)
        t = torch.randint(0, cfg.num_train_timesteps, (x0.shape[0],), device=device)
        noisy = scheduler.add_noise(x0, noise, t)
        target = scheduler.get_velocity(x0, noise, t) if cfg.prediction_type == "v_prediction" else noise

        if conditional:
            labels = labels.to(device)
            drop = torch.rand(labels.shape[0], device=device) < cfg.p_uncond
            labels = torch.where(drop, torch.full_like(labels, cfg.null_class), labels)
        else:
            labels = None

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred = forward_fn(model, noisy, t, labels)
        loss = F.mse_loss(pred.float(), target.float())       # loss in float32 for precision

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update(); lr_sched.step()
        ema.step(model.parameters())

        step += 1
        window.append(loss.item())
        if step % cfg.log_every == 0:
            losses.append(float(np.mean(window))); window = []
            print(f"step {step}/{cfg.total_steps}  loss {losses[-1]:.4f}")
        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            save()
        if step % cfg.milestone_every == 0:
            torch.save({"ema": ema.state_dict(), "step": step, "config": asdict(cfg)},
                       run_dir / f"ema_step{step:06d}.pt")
    return ema, losses


def balanced_labels(cfg: Config, n_per_class: int) -> torch.Tensor:
    """Labels for a class-balanced synthetic set: n_per_class of each class."""
    return torch.arange(cfg.num_classes).repeat_interleave(n_per_class)


@torch.no_grad()
def sample(model, forward_fn, sample_shape, cfg: Config, device, labels=None, n: int | None = None,
           ema=None, guidance: float = 1.0, steps: int | None = None, batch_size: int = 64) -> torch.Tensor:
    """Generate samples with DDIM. Returns a CPU tensor (N, *sample_shape).

    Pass `labels` for conditional generation, or `n` for unconditional.
    guidance=1.0 samples the plain conditional model; use it for every
    distribution-matching metric. Values above 1 trade diversity for fidelity.
    """
    assert (labels is None) != (n is None), "pass exactly one of labels or n"
    total = len(labels) if labels is not None else n
    _, sampler = make_schedulers(cfg)
    sampler.set_timesteps(steps or cfg.inference_steps)

    was_training = model.training
    model.eval()
    if ema is not None:
        ema.store(model.parameters()); ema.copy_to(model.parameters())
    try:
        out = []
        for i in range(0, total, batch_size):
            b = min(batch_size, total - i)
            y = labels[i : i + b].to(device) if labels is not None else None
            x = torch.randn(b, *sample_shape, device=device)
            for t in sampler.timesteps:
                tb = t.expand(b).to(device)
                pred = forward_fn(model, x, tb, y)
                if y is not None and guidance != 1.0:
                    pred_u = forward_fn(model, x, tb, torch.full_like(y, cfg.null_class))
                    pred = pred_u + guidance * (pred - pred_u)
                x = sampler.step(pred, t, x).prev_sample
            out.append(x.cpu())
    finally:
        if ema is not None:
            ema.restore(model.parameters())
        model.train(was_training)
    return torch.cat(out)


def save_samples(file, samples: torch.Tensor, labels, cfg: Config, **meta) -> None:
    """Write generated samples plus the settings that produced them (meta: path name, guidance, ...)."""
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "labels": labels, "config": asdict(cfg), "meta": meta}, file)


def load_samples(file) -> dict:
    return torch.load(file, map_location="cpu")


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------
def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    """Square root of a symmetric positive semi-definite matrix."""
    vals, vecs = np.linalg.eigh(mat)
    return (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T


def frechet_distance(emb_x: np.ndarray, emb_y: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two embedding sets (rows = samples).

    Uses tr sqrt(Sx Sy) = tr sqrt(Sx^1/2 Sy Sx^1/2). The inner matrix is
    symmetric, so the eigendecomposition is valid.
    """
    mu_x, mu_y = emb_x.mean(axis=0), emb_y.mean(axis=0)
    cov_x, cov_y = np.cov(emb_x, rowvar=False), np.cov(emb_y, rowvar=False)
    root_x = _sqrtm_psd(cov_x)
    cross = _sqrtm_psd(root_x @ cov_y @ root_x)
    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(cov_x) + np.trace(cov_y) - 2.0 * np.trace(cross))


@torch.no_grad()
def nearest_neighbor_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """For each query, the L2 distance to its closest item in `bank` (both flattened).

    Memorisation check: compare generated->train against heldout->train. If the
    generated set sits much closer to the training set than real unseen data
    does, the model is copying.
    """
    q, b = queries.flatten(1).float(), bank.flatten(1).float()
    return torch.cat([torch.cdist(q[i : i + chunk], b).min(dim=1).values for i in range(0, len(q), chunk)])
