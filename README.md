# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nimble-condor-29/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nimble-condor-29/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Cohort-maturation forecaster for GTA VI trophy rarity on Exophase: each trophy's earners are the cohorts of tracked players aged through a fitted lognormal earn-delay distribution, forecast against players-tracked growth. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## How it works

A trophy's earn rate is E/N: earners over players tracked. In launch week both move fast, and the rate is not a random walk around its last value: new players join N every hour (diluting the rate) while the players already tracked age into each trophy's delay distribution (lifting E). `main.py` models that mechanism directly, per trophy:

1. **Cohorts** — every read's growth in players-tracked is a cohort that joined at that read's time. The first read's cohort is split 96/4: a small reviewer share with a 120-effective-hour head start (pre-launch play), the rest fresh.
2. **Delay distribution** — a player earns trophy i after a lognormal delay in *effective* hours since launch (weekends count 1.4x). (mu, sigma) are fit per trophy by least squares on log(E+1) over ten recent-weighted reads.
3. **N(target)** — the geometric mean of a logistic fit to players-tracked and constant N: the logistic overshoots while N is still in its exponential phase, constant N undershoots; the blend halves the log error of either.
4. **Forecast** — E(target) by ageing every cohort (history plus future N growth) to the target through the fitted delay distribution; rate = E/N.
5. **Spread** — the pinball-optimal width on a 4.5-day launch horizon is far wider than a random walk's, so sigma is 2.2x the random-walk + fit-residual scale, capped at 4.0.
6. **Story-chain constraints** — chapter rates cannot increase along chapters 1–4 and the final mission; the platinum and 100%-completion trophies cannot exceed any other trophy's rate.

Pure Python 3.13 standard library, deterministic, no network.

## Backtest

Synthetic launch weeks calibrated to the two real anchors available: N=638 at 2026-11-19T02:30Z (the kit's practice history) and N~10000 at 2026-11-20T01:14Z (the live Exophase page). Mean pinball loss **1.21** across 8 worlds and **0.93** across 5 harsher worlds (different delay family, saturation, play rates), against the house persistence baseline's 3.98/3.12 and the house curve baseline's 2.86/1.97 on the same worlds. The backtest harness lives in `work/` (simulator + candidate comparison); `main.py` is the self-contained entry.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=gk6zm613&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
