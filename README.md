# audio-diffusion-synth
Comparing diffusion pipelines for synthetic audio generation, as a testbed for distillation on synthetic data.

## Notebooks (run in Colab, in this order)

| Notebook | What it does |
|---|---|
| `notebooks/path_a_mel_diffusion.ipynb` | Path A: 2D UNet on log-mel spectrograms. Creates the shared `frontend.json`, so run it first. |
| `notebooks/path_b_waveform_diffusion.ipynb` | Path B (raw): 1D UNet on mu-law companded waveforms. |
| `notebooks/path_b_codec_diffusion.ipynb` | Path B (tokenized): 1D UNet on DAC codec token embeddings, decoded back to audio by the codec. |
| `notebooks/path_c_linear_diffusion.ipynb` | Path C: Path A's model on linear-frequency spectrograms. |
| `notebooks/evaluate_paths.ipynb` | Scores every path against held-out real audio in one space: decoded audio, PANNs and CLAP embeddings, representation ceilings, calibrated Frechet and MMD. |

Shared code: `common.py` (config, data, front ends, training loop, sampler) and `eval_metrics.py` (metrics).
