---  
title: Timestral usage notes  
created: 2026-11-27  
tags: [timeseries, forecasting, foundation-models, timestral, reference]  
---  

![png](../../assets/images/2026-11-20-Timestral%20forcaster/backtest.png)
![png](../../assets/images/2026-11-20-Timestral%20forcaster/forecast.png)

## The gist

Mistral launched [Timestral](https://github.com/mistralai/timestral) as a timeseries transformer based forecasting model. They've released **pretrained** models of various sizes. 

You don't train it in a traditional sense, there is no *.fit*  on it. It just one-shot predicts the next values in the series. 

You load the  **frozen weights** from HuggingFace, arrange your historical timeseries data into multiple (B) same-length (T) sequences of the thing you want to predict into a `(B, 1, T)` array and pass that to the model. It returns 32 future steps each with percentile ranges from P1 to P99 `(1, 32, 21)`. 

Most of the time, just the target series isn't enough for accurate predictions so you can pass multiple variables to the forecaster (`(B, V, T)` instead of `(B, 1, T)`) and also pass dependent series that you can compute in advance (day-of-week, holidays, etc.. - called `known_covariates` ) or outside clues (called `past_covariates` ):
* Put a series in targets (`V`) if you want its forecast.
* Use `past_covariates` if it's only a clue and its future is unknown.
* Use `known_covariates` if you know its future values


## Details
  
- `from_pretrained` loads **frozen weights**. `forecast()` runs under `torch.inference_mode()`. No gradients, no optimizer, no per-series training. The repo is inference-only.  
- Your history acts like a **prompt**. The model infers level, scale, seasonality and local dynamics from it, and applies what it learned in pretraining.  
- It is a learned conditional distribution `p(future | history)` evaluated by a network. The model decides in context how much to use a covariate.
- Knobs you actually have: which context you pass, which covariates, and the horizon.  
  
```python  
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")  # device="cuda" for a GPU  
fc = forecaster.forecast(targets, prediction_length=H, past_covariates=past, known_covariates=known)  
```  
  
Model sizes: 5M, 28M, 123M and 209M. All four share the same API, so use a small one to iterate.  
  
## Input shapes  
  
| argument | shape | contents |  
|---|---|---|  
| `context` (targets) | `(B, V, T)` | the series you forecast. `(B, T)` and `(T,)` also work. |  
| `past_covariates` | `(B, C, T)` | clues known **up to the forecast origin only**, same `T` as the targets |  
| `known_covariates` | `(B, K, T + H)` | clues known **in advance**: calendar, holidays, planned events. They must extend `H` steps past the context. |  
  
- `B` = independent series. Rows of different series never interact, so batching is only for speed.  
- `V` = variates forecast **jointly**.  
- Newest observation = **last column**. For batches of different lengths, **pad on the left with NaN** ("right-aligned"). Padding on the right would put the forecast after the padding.  
- **NaN = missing.** The model sees an explicit missing flag.  
- Context is capped at 32,768 steps. The most recent steps are kept.  
  
```python  
targets = df[["n-transactions", "estimated-transaction-volume-usd"]].to_numpy(np.float32).T[None].copy()  # (1, 2, T)  
past    = np.log(df[["market-price"]].to_numpy(np.float32).T[None])                                       # (1, 1, T)  
dates   = pd.date_range(df.index[0], periods=T + H, freq="D")  
known   = np.stack([(dates.day_of_week == d) for d in range(7)]).astype(np.float32)[None]                 # (1, 7, T+H)  
```  

Each row of the batch should be right-aligned. Rows differ in how much history they have, but a batch needs one rectangular array. Here is a toy case with T = 6:

			                 oldest ──────────────────▶ newest (the origin)
			origin A (long)   p1   p2   p3   p4   p5   p6
			origin B (short)  NaN  NaN  p1   p2   p3   p4


> [!warning] Leakage  
> Never put values after the origin in `past_covariates`. A funding rate, an ETF flow or a price on day `t` is known at the end of day `t`. Check that its timestamp is complete before your target's last timestamp.  
  
> [!tip] Backtesting in one call  
> Stack many forecast origins into one batch (`B = n_origins`), left-padded with NaN. Contexts overlap almost entirely. The 7-day forecast windows after each origin are what is non-overlapping. See `stack_contexts` in `crypto_price_forecast.py`. Treat overlapping-context scores as correlated, not as independent samples.  
  
### What a patch is  

Internally, all data is packaged into `patches`.

- A **patch = 32 consecutive time steps of one row** (`patch_size: 32`). It becomes **one token**.  
- Each patch is the 32 values plus 32 "missing" flags, projected to a `d_model` vector (768 for the 209M model). Patches never mix variates.  
- `T = 2048` gives 64 tokens per row.
- Context length should be a multiple of 32, otherwise it is left-padded with unobserved zeros.  
- The horizon is rounded **up** to whole patches. A 7-day or 28-day forecast is computed as 32 days and then trimmed.  
- A 32-day patch contains about 4.6 weeks, so a weekly cycle repeats inside a single token. Interpretation: the model may capture weekly seasonality inside patches rather than by attending to earlier weeks. 

How it's built:
1. A row such as the price, shape (T,), is cut into consecutive chunks of 32: values.reshape(r, n, 32).
2. The matching 32 "missing" flags are cut the same way.
3. Each chunk's 32 values and 32 flags are joined into 64 numbers and passed through a small network. That produces one 768-dimensional vector, the token.

		2048 days of one row:  [d1 … d32][d33 … d64] … [d2017 … d2048]
		                          ↓          ↓               ↓
		                        token 1    token 2   …    token 64


## The forecast predicts 32 steps
  
- The model appends learned **forecast-query tokens**, one per horizon patch. They read the encoded context.  
- A shared head maps each token to **32 steps x 21 quantiles in one pass**. Up to 1,024 steps come out at once.  
  
```python  
# fc.quantiles: (B, V, H, 21), levels 0.01, 0.05, 0.10, 0.15 ... 0.90, 0.95, 0.99  
fc.median[0, v]          # (H,)  
fc.quantile(0.9)[0, v]   # (H,), must be one of fc.quantile_levels  
```  
  
  
## How covariates enter
  
- Covariates are **extra rows** in the same tensor, not a separate input path. Rows are `R = B * (V + C + K)`. Every row gets its own patch tokens through the **same** patch projection.  
- There is **no role embedding**. Targets and covariates differ only through attention masks and scaling.  
- Rows exchange information only in the variate attention inside the `time_variate` layers (3 `time` layers, then 1 `time_variate`, repeated). That attention is restricted to rows of the same series.  
- **One-way firewall** (`variate_firewall`):  
  - A covariate row may not read a target row.  
  - A known-future row may not read non-known rows.  
- **Past-only rows** have a blank horizon. They inform the model over the context only.  
- **Known-future rows** get their future values placed in the window, with non-causal time attention. This is the only way future information reaches the model.  
- Adding any covariate switches from `_predict_context_only` to `_predict_window`. Same weights, different attention layout.  
  
> [!warning] Indicator rows under mirror averaging  
> By default `symmetric_averaging=True`: the model also forecasts the sign-flipped series and averages the two, **negating every covariate row**. Rows containing only 0 and 1 skip normalisation (`instance_norm_stats`). After negation a day-of-week row is `{0, -1}`, no longer recognised as an indicator, so the two passes treat it differently. Effect unmeasured. Test `symmetric_averaging=False` when using indicator covariates.  
  
## Normalisation (why scale doesn't matter)  
  
- **Per row, causal, running.** For each row, a running mean and std over all observed values from the start of the row. Each patch uses the pair at its last step. The first patch is scaled by about 32 days of statistics, and the last context patch by the whole context's.  
- **Why not per dataset:** a zero-shot model cannot know scales in advance (BTC about $60k, DOGE about $0.05, counts about 500k). Dataset-wide or full-series statistics would also leak the future when training causally in parallel. The docstring says it: "the context never sees future statistics."  
- **asinh squash:** `x_norm = asinh((x - mean) / std)`, inverted with `sinh(x)*std + mean`.  
  - `asinh(z)` is about `z` near 0 and logarithmic for large `z`: 3 gives 1.8, 100 gives 5.3, 100000 gives 12.2.  
  - Likely motivation (my reading): heavy tails, and tiny early-series std from the 1e-6 floor.  
  - Both functions are monotone, so quantile ordering survives. `sinh` is exponential, which is why the code has `clamp_nonfinite` and `quantile_real_cap_k`.  
- **Running std = Welford:** `M2 += (x - old_mean) * (x - new_mean)`, `var = M2 / (n - 1)`. This is numerically stable where `E[x^2] - mean^2` is not. The code evaluates it with cumulative sums in float64, no loop.  
- Known-future rows use whole-row (non-causal) mean and std. 0/1 rows skip normalisation.  
- You do **not** need to normalise or difference inputs yourself. In my tests raw vs log targets made no difference. A log is still sensible for a covariate spanning orders of magnitude.  
  
## Using it well: practical checklist  
  
1. Start from the targets alone. Add covariates one at a time and **ablate each** against that baseline.  
2. Compare against a **real baseline** (eg. "Repeat last week").  
3. Report **probabilities from the quantiles**, not a single trajectory.  
4. Keep the scoring scale sane. MASE computed over all history is distorted when the level changed by 100x. Use the trailing year.  



{::options parse_block_html="true" /}
<details><summary markdown='span'>Example</summary>

```python
"""Minimal Timestral example: two targets, a past-only covariate, a known-future covariate.

Forecast Bitcoin daily transactions and USD volume, 28 days ahead.
Data: the CSVs cached in data/ by bitcoin_payments_forecast.py (daily, indexed by date).
"""

import numpy as np
import pandas as pd
from timestral import TimestralForecaster

DATA = "data"          # hardcoded data path
H = 28                 # forecast horizon (days)
T = 2048               # context length (days); the model accepts up to 32768

# ---- Load: one row per calendar day, one column per series -------------------------------
df = pd.concat(
    [pd.read_csv(f"{DATA}/{name}.csv", index_col=0, parse_dates=True)[name]
     for name in ["n-transactions", "estimated-transaction-volume-usd", "market-price"]],
    axis=1, sort=True,
).asfreq("D")                                   # gaps become NaN, which Timestral treats as "missing"
df = df.loc[:df.dropna().index[-1]].tail(T)     # drop a trailing partial day, keep the last T days

# TARGETS (what we forecast).  Shape (B, V, T):
#    B = batch of independent series (rows never interact), V = variates forecast *jointly*, T = time.
#    The newest day must be the LAST column. If you batch histories of different lengths, pad on the
#    LEFT with NaN (right-aligned). Padding on the right would put the forecast after the padding.
targets = df[["n-transactions", "estimated-transaction-volume-usd"]].to_numpy(np.float32).T[None].copy()  # (1, 2, T)

# PAST-ONLY covariates: clues known up to "today" and NOT in the future (price, funding, ...).
#    Shape (B, C, T): same T as the targets. Never put future values here: that would leak the answer.
#    The model sees them only over the context.
past = df[["market-price"]].to_numpy(np.float32).T[None]                         # (1, 1, T)

# KNOWN-FUTURE covariates: clues you know in advance (calendar, holidays, planned events).
#    Shape (B, K, T + H): they must extend H steps PAST the context, because they are the only
#    inputs the model sees over the forecast horizon. Rows containing only 0/1 are treated as indicators.
dates = pd.date_range(df.index[0], periods=T + H, freq="D")
known = np.stack([(dates.day_of_week == d) for d in range(7)]).astype(np.float32)[None]  # (1, 7, T+H)


# ---- Forecast ----------------------------------------------------------------------------
# There is NO fitting here. The weights are frozen and pretrained; `forecast` only runs the network
# on your context, a bit like prompting an LLM. The model infers level, scale and seasonality from the
# context itself, and uses covariates as in-context clues.
#
# Covariates are one-way: targets read them, they never read the targets. The model works in a
# normalised space (rescaled by the context's statistics) and converts the output back to real units.
#
# Mirror averaging: by default (symmetric_averaging=True) the model also forecasts the sign-flipped
#   series and averages the two, negating every covariate row too. That is natural for signed signals,
#   less so for always-positive ones.
forecaster = TimestralForecaster.from_pretrained("mistralai/Timestral-209M")  # device="cuda" for a GPU
fc = forecaster.forecast(targets, prediction_length=H, past_covariates=past, known_covariates=known)

# ---- Read the output ---------------------------------------------------------------------
# fc.quantiles has shape (B, V, H, 21): 21 non-crossing quantile levels per variate and day,
# 0.01, 0.05, 0.10, 0.15, ... 0.90, 0.95, 0.99 (see fc.quantile_levels). Use the spread, not just
# the median: it is a calibrated range (P10-P90 should contain ~80% of actuals).
future = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=H, freq="D")
for v, name in enumerate(["n-transactions", "volume-usd"]):
    out = pd.DataFrame({"p10": fc.quantile(0.1)[0, v], "median": fc.median[0, v], "p90": fc.quantile(0.9)[0, v]},
                       index=future)
    print(f"\n{name}\n{out.head(7).to_string(float_format='{:,.0f}'.format)}")
```
</details>
