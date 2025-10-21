# pm_lstm_gru

Real‑time, causal LSTM/GRU pipeline to predict PM10/PM2.5/PM1 from vehicle dynamics.

## Quick start (WSL + pixi)
```bash
# 1) In WSL
cd ~/Projects && mkdir -p pm_lstm_gru && cd pm_lstm_gru

# 2) Initialize environment
pixi init      # creates pixi.toml if not present
# If pixi.toml already provided, just:
pixi install

# 3) Put your CSV(s)
mkdir -p data/raw && cp /mnt/c/Path/To/1.csv data/raw/

# 4) Train
pixi run train
```

## What this repo guarantees
- **Causality**: all transforms (scaling, EMA, rolling stats) fit/update on past data only.
- **Leak‑proof splits**: walk‑forward CV with purge windows around boundaries; per‑trip grouping.
- **Event awareness**: idle/brake/cruise regimes for balanced training and metrics.
- **Two backbones**: LSTM and GRU behind a shared interface; optional Conv1d front‑end.

## Config knobs
See `src/config.py` for: input window, stride, target shift τ (default 20 s), downsampling (5 Hz), model type (`lstm`|`gru`), hidden sizes, dropout, loss weights, and sampler weights.

## Outputs
- Checkpoints in `.runs/` (best by event‑weighted MAE)
- Plots in `figures/` (residuals, regime MAE, prediction overlays)
- CSVs of metrics per fold and per regime
