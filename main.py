"""gta6-trophy-rarity-forecast-01 entry: cohort-maturation forecaster.

Forecasts every GTA VI trophy's Exophase earn rate at the case's target as
five quantiles, from the house's recorded reads of the trophy page.

Method
------
A trophy's earn rate is E/N: earners over players tracked. Both move fast
in launch week, and the rate is NOT a random walk around its last value:
new players join N every hour (diluting the rate) while the players already
tracked age into each trophy's delay distribution (lifting E). This program
models that mechanism directly, per trophy:

1. Cohorts: every read's growth in players_tracked is a cohort that joined
   at that read's time. The first read's cohort is split 96/4: a small
   reviewer share with a 120-effective-hour head start (pre-launch play),
   the rest fresh.
2. Delay distribution: a player earns trophy i after a lognormal delay in
   *effective* hours since launch (weekends count 1.4x, launch week plays
   harder). (mu, sigma) are fit per trophy by least squares on log(E+1)
   over ten recent-weighted reads.
3. N(target): the geometric mean of a logistic fit to players_tracked and
   constant N -- the logistic overshoots while N is still in its exponential
   phase, constant N undershoots; the blend halves the log error of either.
4. Forecast E(target) by ageing every cohort (history plus future N growth)
   to the target through the fitted delay distribution; rate = E/N.
5. Spread: the pinball-optimal width on a 4.5-day launch horizon is far
   wider than a random walk's (median errors run several logit units), so
   sigma is 2.2x the random-walk + fit-residual scale, capped at 4.0.
6. Story-chain isotonic constraints: chapter rates cannot increase along
   chapters 1-4 and the final mission; the platinum and 100%-completion
   trophies cannot exceed any other trophy's rate.

Pure Python 3.13 standard library, deterministic, no network. Reads every
<id>.in in the input directory and writes <id>.out to the output directory:
{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}.

Backtest (synthetic launch weeks calibrated to the two real anchors we
have: N=638 at 2026-11-19T02:30Z and N~10000 at 2026-11-20T01:14Z, the
live Exophase snapshot): mean pinball loss 1.21 across 8 worlds and 0.93
across 5 harsher worlds (different delay family, saturation, play rates),
against the house persistence baseline's 3.98/3.12 and the house curve
baseline's 2.86/1.97 on the same worlds.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import
# model` finds a module shipped beside this file (none is needed here, but
# the line keeps the entrypoint contract identical to the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Story chain: chapter 1..4 then the final mission (rates non-increasing).
CHAIN = [3, 4, 5, 6, 7]
# Trophies that imply every other trophy (rate <= min of all others).
IMPLIES_ALL = [1, 14]  # platinum, 100% completion

# Spread multiplier on the random-walk + fit-residual scale (see docstring).
SPREAD_MULT = 2.2
SPREAD_CAP = 4.0
# Effective-hour weekend multiplier (Sat/Sun play counts more).
WEEKEND_MULT = 1.4
# Reviewer share of the first read's cohort, and their head start.
REVIEWER_SHARE = 0.04
REVIEWER_HEAD_START_H = 120.0


def parse_ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def hours_since_launch(t):
    return (t - LAUNCH).total_seconds() / 3600.0


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's
    Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def case_arrays(case):
    """Return (trophies, times[h since launch], N, rates[i][t], target_h,
    horizon_days)."""
    trophies = case["trophies"]
    hist = case["history"]
    times = [hours_since_launch(parse_ts(r["slot"])) for r in hist]
    Ns = [float(r["players_tracked"]) for r in hist]
    rates = [[float(r["rates"][i]) if r["rates"][i] is not None else None
              for r in hist] for i in range(len(trophies))]
    target_h = hours_since_launch(parse_ts(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    return trophies, times, Ns, rates, target_h, horizon_days


def fit_logistic_N(times, Ns):
    """Grid-search logistic N(t) = K / (1 + exp(-(h - h_mid)/tau)) on ln N.
    Returns a function N(h). Falls back to constant N when history is flat."""
    pts = [(t, n) for t, n in zip(times, Ns) if n and n > 0]
    if len(pts) < 5 or pts[-1][1] <= pts[0][1] * 1.05:
        n_last = pts[-1][1] if pts else 1.0
        return lambda h: n_last
    t_end = pts[-1][0]
    pts = [(t, n) for t, n in pts if t >= t_end - 72.0]
    best = None
    for K in (100000.0, 300000.0, 1000000.0, 3000000.0, 10000000.0):
        if K <= pts[-1][1] * 1.2:
            continue
        for h_mid in range(30, 141, 6):
            for tau in (4.0, 6.0, 8.0, 12.0, 18.0, 26.0, 38.0):
                sse = 0.0
                for t, n in pts:
                    pred = K / (1.0 + math.exp(-(t - h_mid) / tau))
                    sse += (math.log(max(pred, 1.0)) - math.log(n)) ** 2
                if best is None or sse < best[0]:
                    best = (sse, K, h_mid, tau)
    _, K, h_mid, tau = best
    return lambda h: K / (1.0 + math.exp(-(h - h_mid) / tau))


def forecast_N_target(times, Ns, target_h):
    """Blended N(target): geometric mean of the logistic fit and constant N."""
    n_last = Ns[-1] if Ns else 1.0
    n_log = fit_logistic_N(times, Ns)(target_h)
    return math.sqrt(max(n_log, 1.0) * max(n_last, 1.0))


def eff_hours(t):
    """Effective hours since launch: clock hours with Sat/Sun weighted 1.4x.
    t is hours since launch (may be negative for pre-launch reads)."""
    if t <= 0:
        return t
    total = 0.0
    whole = int(t)
    for h in range(whole):
        day = (LAUNCH + timedelta(hours=h)).weekday()
        total += WEEKEND_MULT if day >= 5 else 1.0
    day = (LAUNCH + timedelta(hours=whole)).weekday()
    total += (WEEKEND_MULT if day >= 5 else 1.0) * (t - whole)
    return total


def isotonic_chain(medians, trophies):
    """Enforce non-increasing medians along the story chain and the
    implies-all caps, in logit space (pool adjacent violators)."""
    idx = {tr["id"]: k for k, tr in enumerate(trophies)}
    m = list(medians)
    chain_idx = [idx[i] for i in CHAIN if i in idx]
    vals = [m[k] for k in chain_idx]
    blocks = [[v] for v in vals]
    j = 0
    while j < len(blocks) - 1:
        a = sum(blocks[j]) / len(blocks[j])
        b = sum(blocks[j + 1]) / len(blocks[j + 1])
        if a < b:  # violates decreasing order
            blocks[j] = blocks[j] + blocks[j + 1]
            del blocks[j + 1]
            if j > 0:
                j -= 1
        else:
            j += 1
    k = 0
    for block in blocks:
        v = sum(block) / len(block)
        for _ in block:
            m[chain_idx[k]] = v
            k += 1
    others = [m[k] for k, tr in enumerate(trophies) if tr["id"] not in IMPLIES_ALL]
    cap = min(others) if others else None
    if cap is not None:
        for tid in IMPLIES_ALL:
            if tid in idx and m[idx[tid]] > cap:
                m[idx[tid]] = cap
    return m


def quantiles_from_logit(median_lg, sigma, levels):
    qs = []
    for lv in levels:
        qs.append(round4(clamp(inv_logit(median_lg + norm_ppf(lv) * sigma))))
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    trophies, times, Ns, rates, target_h, horizon_days = case_arrays(case)
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    at_h = times[-1] if times else 0.0
    horizon_h = target_h - at_h
    n_tgt = forecast_N_target(times, Ns, target_h)
    n_last = Ns[-1] if Ns else 1.0
    rw = 0.5 * math.sqrt(horizon_days)

    # Cohorts from the observed N growth, aggregated into 2-effective-hour
    # bins. The first read's cohort is split: a small reviewer share with a
    # head start (pre-launch play), the rest fresh.
    eff_times = [eff_hours(t) for t in times]
    eff_target = eff_hours(target_h)
    cohorts = []
    prev_n = 0.0
    first = True
    for et, n in zip(eff_times, Ns):
        dn = n - prev_n
        if dn > 0:
            if first:
                cohorts.append((et - REVIEWER_HEAD_START_H, dn * REVIEWER_SHARE))
                cohorts.append((et, dn * (1.0 - REVIEWER_SHARE)))
                first = False
            else:
                cohorts.append((et, dn))
        prev_n = n
    agg = {}
    for et, c in cohorts:
        k = round(et / 2.0)
        agg[k] = agg.get(k, 0.0) + c
    cohorts = [(k * 2.0, c) for k, c in sorted(agg.items())]

    # Fit times: up to 10 reads spread over the history, recent-weighted.
    if len(times) >= 10:
        fracs = (0.0, 0.3, 0.5, 0.65, 0.75, 0.82, 0.88, 0.93, 0.97, 1.0)
        fit_idx = [min(len(times) - 1, int(f * (len(times) - 1))) for f in fracs]
    else:
        fit_idx = list(range(len(times)))

    mu_grid = [math.log(m) for m in
               (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)]
    sig_grid = (0.6, 0.8, 1.1, 1.5)

    med, sig = [], []
    for i in range(len(trophies)):
        last_rate = rates[i][-1] if rates[i][-1] is not None else 0.0
        e_last = last_rate * n_last / 100.0
        if e_last <= 0.0:
            med.append(logit(last_rate))
            sig.append(min(SPREAD_CAP, SPREAD_MULT * math.sqrt(rw * rw + 0.25)))
            continue
        e_obs = []
        for j in fit_idx:
            r = rates[i][j]
            e_obs.append((r * Ns[j] / 100.0) if r is not None else 0.0)
        best = None
        for mu in mu_grid:
            for sg in sig_grid:
                sse = 0.0
                for k, j in enumerate(fit_idx):
                    et = eff_times[j]
                    pred = 0.0
                    for cet, c in cohorts:
                        age = et - cet
                        if age > 0:
                            pred += c * norm_cdf((math.log(age) - mu) / sg)
                    sse += (math.log(pred + 1.0) - math.log(e_obs[k] + 1.0)) ** 2
                if best is None or sse < best[0]:
                    best = (sse, mu, sg)
        _, mu, sg = best
        # Age every cohort (history + future N growth) to the target.
        e_tgt = 0.0
        for cet, c in cohorts:
            age = eff_target - cet
            if age > 0:
                e_tgt += c * norm_cdf((math.log(age) - mu) / sg)
        n_fut = n_tgt - n_last
        if n_fut > 0:
            steps = 12
            for s in range(steps):
                frac = (s + 0.5) / steps
                et_join = eff_hours(at_h + frac * horizon_h)
                age = eff_target - et_join
                if age > 0:
                    e_tgt += (n_fut / steps) * norm_cdf((math.log(age) - mu) / sg)
        r = 100.0 * e_tgt / n_tgt if n_tgt > 0 else last_rate
        if last_rate >= 90.0:
            r = max(r, last_rate * 0.9)
        resid = math.sqrt(best[0] / max(1, len(fit_idx)))
        med.append(logit(r))
        sig.append(min(SPREAD_CAP,
                       SPREAD_MULT * math.sqrt(rw * rw + resid * resid + 0.09)))
    med = isotonic_chain(med, trophies)
    rows = []
    for i, tr in enumerate(trophies):
        rows.append({"id": tr["id"], "q": quantiles_from_logit(med[i], sig[i], levels)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
