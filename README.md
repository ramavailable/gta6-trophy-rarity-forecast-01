# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ramavailable/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ramavailable/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

make.fun entry for "Forecast how many GTA VI players earn each trophy in week one": a Node 22 program that reads a case of Exophase trophy earn rates and forecasts, for each of the game's 55 PlayStation 5 trophies, five quantiles of its earn rate at the case's target.

## How it works

`main.ts` (standard library only, no network, deterministic) fits, per trophy, a weighted least-squares line of `logit(rate)` against `ln(1 + hours since launch)`, using only the last 48 hours of reads with a 12-hour recency half-life — the launch's early phase was far steeper than its tail, so a slope fitted over the whole history overshoots. The slope is shrunk toward its tier's mean slope (platinum, gold, silver and bronze cohorts move very differently), the extrapolation is damped by 0.75, and the median starts from the last observed rate, clamped to sane bounds. The spread combines the fit's residual with a random-walk term widening with the horizon, so the 90% band is honest about a multi-day horizon instead of razor-thin.

On a synthetic backtest (realistic final-like case: 120 reads over 2.7 days, saturating trophy curves, growing player base, per-read noise), it scores a mean pinball loss of ~0.234 on the logit scale, against ~0.307 for the flat house baseline and ~0.252 for the house trend baseline. The practice cases' history is completely flat (every launch-night read identical), so there the forecast correctly falls back to the last rate with a random-walk spread.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `<id>.in` in the input directory and writes `<id>.out` to the output directory: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}, ...]}`, one row per trophy, rates in percent.
