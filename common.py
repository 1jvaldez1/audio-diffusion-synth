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
    total_steps: int = 20000
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

    def take(mask):
        return {
            "waveforms": data["waveforms"][mask],
            "labels": data["labels"][mask],
            "folds": data["folds"][mask],
            "class_names": data["class_names"],
        }

    return take(~held), take(held)


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


def reference_crops(split: dict, cfg: Config, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic crops for evaluation: up to k valid crops per clip, evenly spread.

    Use on the held-out split for the Frechet reference, and on the train split
    for the memorisation check.
    """
    starts, valid = valid_crop_starts(split["waveforms"], cfg)
    L, crops, labels = cfg.clip_samples, [], []
    for i, v in enumerate(valid):
        idx = torch.nonzero(v).squeeze(1)
        pick = idx[torch.linspace(0, len(idx) - 1, k).round().long().unique()]
        for s in starts[pick].tolist():
            crops.append(split["waveforms"][i, :, s : s + L])
            labels.append(split["labels"][i])
    return torch.stack(crops), torch.stack(labels)


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
