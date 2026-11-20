"""Candidate forecasters for gta6-trophy-rarity-forecast-01.

All operate on a case dict in the kit's schema (trophies, history reads with
slot/at/players_tracked/rates, horizon_minutes, target, levels) and return
{"trophies": [{"id": int, "q": [q05, q25, q50, q75, q95]}, ...]}.

Models
------
- persist: house persistence baseline (starter replication).
- curve: house curve baseline replication: OLS of logit(rate) on
  ln(1 + hours since 2026-11-19T00:00Z) over the last 24h of reads.
- en_power: E/N decomposition with power-law (log-time) fits of E and N.
- hazard: E/N decomposition with a per-trophy earn hazard integrated
  forward against a logistic extrapolation of N(t).
- ensemble: logit-space weighted blend + story-chain isotonic constraints.

Pure standard library; the chosen model will be vendored into main.py.
"""
import math
from datetime import datetime, timedelta, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Story chain: chapter 1..4 then final mission (rates must not increase).
CHAIN = [3, 4, 5, 6, 7]
# Trophies that imply every other trophy (rate <= min of all others).
IMPLIES_ALL = [1, 14]  # platinum, 100% completion


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
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def ols(xs, ys):
    """Least squares y ~ a + b x. Returns (a, b, resid_std)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0, 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2))
    else:
        resid = 0.0
    return (a, b, resid)


def case_arrays(case):
    """Return (trophies, times[h since launch], N, rates[i][t], target_h, horizon_days)."""
    trophies = case["trophies"]
    hist = case["history"]
    times = [hours_since_launch(parse_ts(r["slot"])) for r in hist]
    Ns = [float(r["players_tracked"]) for r in hist]
    rates = [[float(r["rates"][i]) if r["rates"][i] is not None else None
              for r in hist] for i in range(len(trophies))]
    target_h = hours_since_launch(parse_ts(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    return trophies, times, Ns, rates, target_h, horizon_days


def quantiles_from_logit(median_lg, sigma):
    qs = []
    for z in Z:
        qs.append(round4(clamp(inv_logit(median_lg + z * sigma))))
    # enforce non-decreasing (rounding can tie, never invert, but be safe)
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def make_rows(trophies, medians_lg, sigmas):
    rows = []
    for i, tr in enumerate(trophies):
        rows.append({"id": tr["id"], "q": quantiles_from_logit(medians_lg[i], sigmas[i])})
    return {"trophies": rows}


# ---------------------------------------------------------------- baselines
def persist(case):
    _, _, _, rates, _, horizon_days = case_arrays(case)
    spread = 0.5 * math.sqrt(horizon_days)
    med, sig = [], []
    for i in range(len(case["trophies"])):
        last = rates[i][-1]
        last = 0.0 if last is None else last
        med.append(logit(last))
        sig.append(spread)
    return make_rows(case["trophies"], med, sig)


def curve(case):
    """House curve baseline replication."""
    trophies, times, _, rates, target_h, horizon_days = case_arrays(case)
    at_h = times[-1] if times else 0.0
    xt = math.log1p(target_h)
    rw = 0.5 * math.sqrt(horizon_days)
    med, sig = [], []
    for i in range(len(trophies)):
        xs, ys = [], []
        for j, t in enumerate(times):
            if t >= at_h - 24.0 and rates[i][j] is not None:
                xs.append(math.log1p(t))
                ys.append(logit(rates[i][j]))
        if len(xs) >= 3:
            a, b, resid = ols(xs, ys)
            m = a + b * xt
            s = math.sqrt(rw * rw + resid * resid)
        else:
            m = logit(rates[i][-1] if rates[i][-1] is not None else 0.0)
            s = rw
        med.append(m)
        sig.append(s)
    return make_rows(trophies, med, sig)


def _curve_components(case):
    """Return per-trophy (median_logit, resid) of the curve fit, for blending."""
    trophies, times, _, rates, target_h, _ = case_arrays(case)
    at_h = times[-1] if times else 0.0
    xt = math.log1p(target_h)
    out = []
    for i in range(len(trophies)):
        xs, ys = [], []
        for j, t in enumerate(times):
            if t >= at_h - 24.0 and rates[i][j] is not None:
                xs.append(math.log1p(t))
                ys.append(logit(rates[i][j]))
        if len(xs) >= 3:
            a, b, resid = ols(xs, ys)
            out.append((a + b * xt, resid))
        else:
            out.append((logit(rates[i][-1] if rates[i][-1] is not None else 0.0), 0.0))
    return out


# ------------------------------------------------------- N(t) logistic fit
def fit_logistic_N(times, Ns):
    """Grid-search logistic N(t) = K / (1 + exp(-(h - h_mid)/tau)) on ln N.
    Returns a function N(h). Falls back to constant N when history is flat."""
    pts = [(t, n) for t, n in zip(times, Ns) if n and n > 0]
    if len(pts) < 5 or pts[-1][1] <= pts[0][1] * 1.05:
        n_last = pts[-1][1] if pts else 1.0
        return lambda h: n_last
    # use the last 72h of reads
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
    """Blended N(target): geometric mean of the logistic fit and constant N.
    The logistic overshoots while N is still in its exponential phase and
    constant N undershoots; the blend halves the log error of either."""
    n_last = Ns[-1] if Ns else 1.0
    n_log = fit_logistic_N(times, Ns)(target_h)
    return math.sqrt(max(n_log, 1.0) * max(n_last, 1.0))


def fit_power_N(times, Ns, target_h):
    """Power-law fit ln N ~ a + b ln(1+h) over the last 48h; extrapolate."""
    t_end = times[-1] if times else 0.0
    xs, ys = [], []
    for t, n in zip(times, Ns):
        if t >= t_end - 48.0 and n and n > 0:
            xs.append(math.log1p(t))
            ys.append(math.log(n))
    if len(xs) < 3:
        return Ns[-1] if Ns else 1.0
    a, b, _ = ols(xs, ys)
    return math.exp(a + b * math.log1p(target_h))


# ------------------------------------------------------------- E/N models
def en_power(case):
    """Forecast E_i and N separately with log-time power laws, rate = E/N."""
    trophies, times, Ns, rates, target_h, horizon_days = case_arrays(case)
    at_h = times[-1] if times else 0.0
    xt = math.log1p(target_h)
    n_tgt = fit_power_N(times, Ns, target_h)
    rw = 0.5 * math.sqrt(horizon_days)
    med, sig = [], []
    for i in range(len(trophies)):
        xs, ys = [], []
        for j, t in enumerate(times):
            if t >= at_h - 48.0 and rates[i][j] is not None and Ns[j] > 0:
                e = rates[i][j] * Ns[j] / 100.0
                if e >= 2.0:
                    xs.append(math.log1p(t))
                    ys.append(math.log(e))
        if len(xs) >= 3:
            a, b, resid = ols(xs, ys)
            e_tgt = math.exp(a + b * xt)
            r = 100.0 * e_tgt / n_tgt if n_tgt > 0 else rates[i][-1]
            m = logit(r)
            s = math.sqrt(rw * rw + resid * resid)
        else:
            m = logit(rates[i][-1] if rates[i][-1] is not None else 0.0)
            s = rw
        med.append(m)
        sig.append(s)
    return make_rows(trophies, med, sig)


def _en_power_components(case):
    trophies, times, Ns, rates, target_h, _ = case_arrays(case)
    at_h = times[-1] if times else 0.0
    xt = math.log1p(target_h)
    n_tgt = fit_power_N(times, Ns, target_h)
    out = []
    for i in range(len(trophies)):
        xs, ys = [], []
        for j, t in enumerate(times):
            if t >= at_h - 48.0 and rates[i][j] is not None and Ns[j] > 0:
                e = rates[i][j] * Ns[j] / 100.0
                if e >= 2.0:
                    xs.append(math.log1p(t))
                    ys.append(math.log(e))
        if len(xs) >= 3:
            a, b, resid = ols(xs, ys)
            e_tgt = math.exp(a + b * xt)
            r = 100.0 * e_tgt / n_tgt if n_tgt > 0 else rates[i][-1]
            out.append((logit(r), resid))
        else:
            out.append((logit(rates[i][-1] if rates[i][-1] is not None else 0.0), 0.0))
    return out


def hazard(case):
    """Per-trophy earn hazard from the last 24h, integrated forward against
    a logistic N(t); near-100% trophies are modelled on the complement."""
    trophies, times, Ns, rates, target_h, horizon_days = case_arrays(case)
    at_h = times[-1] if times else 0.0
    horizon_h = target_h - at_h
    n_tgt = forecast_N_target(times, Ns, target_h)
    n_last = Ns[-1] if Ns else 1.0

    def n_path(h):
        if horizon_h <= 0:
            return n_last
        f = (h - at_h) / horizon_h
        f = min(1.0, max(0.0, f))
        return n_last * (n_tgt / n_last) ** f if n_last > 0 else n_last

    rw = 0.5 * math.sqrt(horizon_days)
    med, sig = [], []
    for i in range(len(trophies)):
        # series of E
        es = []
        for j, t in enumerate(times):
            if rates[i][j] is None or Ns[j] <= 0:
                es.append(None)
            else:
                es.append(rates[i][j] * Ns[j] / 100.0)
        last_rate = rates[i][-1] if rates[i][-1] is not None else 0.0
        last_e = es[-1] if es[-1] is not None else 0.0
        last_n = Ns[-1] if Ns else 1.0
        # hazard over the last 24h: new earners / avg susceptible / hours
        j0 = 0
        for j, t in enumerate(times):
            if t >= at_h - 24.0:
                j0 = j
                break
        e0 = es[j0] if es[j0] is not None else last_e
        n0 = Ns[j0] if j0 < len(Ns) else last_n
        dt = max(1.0, at_h - times[j0])
        if last_rate >= 90.0:
            # model the complement U = N - E
            u_last = max(0.0, last_n - last_e)
            u0 = max(0.0, n0 - e0)
            # complement grows with N and shrinks as players earn it:
            # assume U scales with N at the recent ratio, damped
            ratio = (u_last / last_n) if last_n > 0 else 0.0
            u_tgt = ratio * n_tgt * 0.85  # slight earn-through
            r = 100.0 * (1.0 - u_tgt / n_tgt) if n_tgt > 0 else last_rate
            med.append(logit(r))
            sig.append(math.sqrt(rw * rw + 0.25))
            continue
        susc = max(1.0, 0.5 * ((n0 - e0) + (last_n - last_e)))
        new_e = max(0.0, last_e - e0)
        haz = (new_e / susc / dt) if susc > 0 else 0.0
        haz = min(haz, 0.05)  # cap: at most 5% of susceptibles per hour
        # integrate forward on an hourly grid with N(t) from the fit
        e_cur = last_e
        steps = max(1, int(round(horizon_h)))
        for s in range(steps):
            h = at_h + (s + 0.5) * horizon_h / steps
            n_h = n_path(h)
            susc_h = max(0.0, n_h - e_cur)
            e_cur += haz * 0.9 * susc_h * (horizon_h / steps)
        r = 100.0 * e_cur / n_tgt if n_tgt > 0 else last_rate
        # residual scale: variability of hourly hazard (rough proxy)
        med.append(logit(r))
        sig.append(math.sqrt(rw * rw + 0.09))
    return make_rows(trophies, med, sig)


def _hazard_components(case):
    """Same as hazard() but returns per-trophy (median_logit, sigma)."""
    out = hazard(case)
    rows = out["trophies"]
    res = []
    for row in rows:
        q = row["q"]
        res.append((logit(q[2]), 0.0))
    return res


# ------------------------------------------------- cohort-maturation model
def _eff_hours(t):
    """Effective clock hours since launch, with Sat/Sun play boosted 1.4x.
    t is hours since launch (float, may be negative for pre-launch reads)."""
    if t <= 0:
        return t
    # integrate the weekend multiplier on an hourly grid
    total = 0.0
    whole = int(t)
    for h in range(whole):
        day = (datetime(2026, 11, 19, tzinfo=timezone.utc)
               + timedelta(hours=h)).weekday()
        total += 1.4 if day >= 5 else 1.0
    h = whole
    day = (datetime(2026, 11, 19, tzinfo=timezone.utc)
           + timedelta(hours=h)).weekday()
    total += (1.4 if day >= 5 else 1.0) * (t - whole)
    return total


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def cohort(case):
    """Delay-distribution model: each trophy's earners are cohorts that joined
    N at time s and earn after a lognormal delay in effective hours.
    (mu, sigma) are fit per trophy to the observed E(t); the forecast ages
    every cohort (history + future N growth) to the target."""
    from datetime import timedelta
    trophies, times, Ns, rates, target_h, horizon_days = case_arrays(case)
    at_h = times[-1] if times else 0.0
    horizon_h = target_h - at_h
    n_tgt = forecast_N_target(times, Ns, target_h)
    n_last = Ns[-1] if Ns else 1.0
    rw = 0.5 * math.sqrt(horizon_days)
    # cohorts: (effective join time, size). Aggregate reads into 2h bins.
    eff_times = [_eff_hours(t) for t in times]
    eff_target = _eff_hours(target_h)
    cohorts = []
    prev_n = 0.0
    first = True
    for j, (et, n) in enumerate(zip(eff_times, Ns)):
        dn = n - prev_n
        if dn > 0:
            if first:
                # initial cohort: 4% reviewers with a 120h effective head start
                cohorts.append((et - 120.0, dn * 0.04))
                cohorts.append((et, dn * 0.96))
                first = False
            else:
                cohorts.append((et, dn))
        prev_n = n
    # aggregate cohorts into 2-effective-hour bins to keep the fit fast
    agg = {}
    for et, c in cohorts:
        k = round(et / 2.0)
        agg[k] = agg.get(k, 0.0) + c
    cohorts = [(k * 2.0, c) for k, c in sorted(agg.items())]
    # fit times: up to 10 reads spread over the history, recent-weighted
    fit_idx = []
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
        e_obs = []
        for j in fit_idx:
            r = rates[i][j]
            e_obs.append((r * Ns[j] / 100.0) if r is not None else 0.0)
        e_last = rates[i][-1] * n_last / 100.0 if rates[i][-1] is not None else 0.0
        if e_last <= 0.0:
            med.append(logit(rates[i][-1] if rates[i][-1] is not None else 0.0))
            sig.append(min(4.0, 2.2 * math.sqrt(rw * rw + 0.25)))
            continue
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
                            pred += c * _norm_cdf((math.log(age) - mu) / sg)
                    sse += (math.log(pred + 1.0) - math.log(e_obs[k] + 1.0)) ** 2
                if best is None or sse < best[0]:
                    best = (sse, mu, sg)
        _, mu, sg = best
        # forecast: age every cohort (history + future) to the target
        e_tgt = 0.0
        for cet, c in cohorts:
            age = eff_target - cet
            if age > 0:
                e_tgt += c * _norm_cdf((math.log(age) - mu) / sg)
        # future cohorts: N grows from n_last to n_tgt, spread over the window
        n_fut = n_tgt - n_last
        if n_fut > 0:
            steps = 12
            for s in range(steps):
                frac = (s + 0.5) / steps
                et_join = _eff_hours(at_h + frac * horizon_h)
                age = eff_target - et_join
                if age > 0:
                    e_tgt += (n_fut / steps) * _norm_cdf((math.log(age) - mu) / sg)
        r = 100.0 * e_tgt / n_tgt if n_tgt > 0 else rates[i][-1]
        # near-100% trophies: the delay fit saturates; keep the complement sane
        if rates[i][-1] is not None and rates[i][-1] >= 90.0:
            r = max(r, rates[i][-1] * 0.9)
        resid = math.sqrt(best[0] / max(1, len(fit_idx)))
        med.append(logit(r))
        # Spread: the backtest's optimal width is ~2.2x the naive
        # random-walk + fit-residual scale (median errors run several logit
        # units on a 4.5-day launch horizon); cap it so quantiles stay sane.
        sig.append(min(4.0, 2.2 * math.sqrt(rw * rw + resid * resid + 0.09)))
    med = _isotonic_chain(med, trophies)
    return make_rows(trophies, med, sig)


def _cohort_medlogit(case):
    return [logit(row["q"][2]) for row in cohort(case)["trophies"]]


def blend2(case, w_cohort=0.7, w_haz=0.3, chain=True):
    """Blend of the cohort-maturation model and the hazard model."""
    trophies = case["trophies"]
    co = _cohort_medlogit(case)
    hz = _rows_medlogit(hazard(case)["trophies"])
    med = [w_cohort * a + w_haz * b for a, b in zip(co, hz)]
    if chain:
        med = _isotonic_chain(med, trophies)
    _, _, _, _, _, horizon_days = case_arrays(case)
    rw = 0.5 * math.sqrt(horizon_days)
    sig = [math.sqrt(rw * rw + 0.25)] * len(trophies)
    return make_rows(trophies, med, sig)


# --------------------------------------------------------------- ensemble
def _rows_medlogit(rows):
    return [logit(row["q"][2]) for row in rows]


def _hazard_sigma(case):
    _, _, _, _, _, horizon_days = case_arrays(case)
    rw = 0.5 * math.sqrt(horizon_days)
    return [math.sqrt(rw * rw + 0.09)] * len(case["trophies"])


def hazard_chain(case):
    """Hazard model + story-chain isotonic constraints."""
    trophies = case["trophies"]
    med = _rows_medlogit(hazard(case)["trophies"])
    med = _isotonic_chain(med, trophies)
    return make_rows(trophies, med, _hazard_sigma(case))


def blend(case, w_haz=0.7, w_curve=0.3, w_persist=0.0, chain=True,
          sigma_extra=0.0):
    """Logit-space blend of hazard / curve / persistence medians."""
    trophies = case["trophies"]
    haz = _rows_medlogit(hazard(case)["trophies"])
    cur = [m for m, _ in _curve_components(case)]
    per = [logit(r) for r in _last_rates(case)]
    med = []
    for i in range(len(trophies)):
        med.append(w_haz * haz[i] + w_curve * cur[i] + w_persist * per[i])
    if chain:
        med = _isotonic_chain(med, trophies)
    sig = _hazard_sigma(case)
    if sigma_extra:
        sig = [math.sqrt(s * s + sigma_extra ** 2) for s in sig]
    return make_rows(trophies, med, sig)


def _last_rates(case):
    _, _, _, rates, _, _ = case_arrays(case)
    return [rates[i][-1] if rates[i][-1] is not None else 0.0
            for i in range(len(case["trophies"]))]


def _isotonic_chain(medians, trophies):
    """Enforce non-increasing medians along the story chain and the
    implies-all caps, in logit space. medians: list aligned with trophies."""
    idx = {tr["id"]: k for k, tr in enumerate(trophies)}
    m = list(medians)
    # chain: pool adjacent violators (non-increasing)
    chain_idx = [idx[i] for i in CHAIN if i in idx]
    vals = [m[k] for k in chain_idx]
    # PAVA for decreasing sequence
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
    # implies-all: platinum and 100% <= min of all other medians
    others = [m[k] for k, tr in enumerate(trophies) if tr["id"] not in IMPLIES_ALL]
    cap = min(others) if others else None
    if cap is not None:
        for tid in IMPLIES_ALL:
            if tid in idx and m[idx[tid]] > cap:
                m[idx[tid]] = cap
    return m


def ensemble(case, w_persist=0.15, w_curve=0.30, w_en=0.20, w_haz=0.35,
             sigma_floor=0.35, sigma_cap=2.5):
    trophies, times, Ns, rates, target_h, horizon_days = case_arrays(case)
    rw = 0.5 * math.sqrt(horizon_days)
    persist_med = [logit(rates[i][-1] if rates[i][-1] is not None else 0.0)
                   for i in range(len(trophies))]
    curve_med = [m for m, _ in _curve_components(case)]
    curve_res = [r for _, r in _curve_components(case)]
    en_med = [m for m, _ in _en_power_components(case)]
    en_res = [r for _, r in _en_power_components(case)]
    haz_med = [m for m, _ in _hazard_components(case)]
    # flat history -> all components agree with persistence; blending is safe
    med, sig = [], []
    for i in range(len(trophies)):
        comps = [persist_med[i], curve_med[i], en_med[i], haz_med[i]]
        w = [w_persist, w_curve, w_en, w_haz]
        m = sum(wi * ci for wi, ci in zip(w, comps))
        disag = math.sqrt(sum(wi * (ci - m) ** 2 for wi, ci in zip(w, comps)))
        s = math.sqrt(rw * rw + curve_res[i] ** 2 + en_res[i] ** 2 + disag ** 2)
        s = min(sigma_cap, max(sigma_floor, s))
        med.append(m)
        sig.append(s)
    med = _isotonic_chain(med, trophies)
    return make_rows(trophies, med, sig)
