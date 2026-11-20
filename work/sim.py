"""Simulator for GTA VI trophy-rarity launch dynamics + pinball scorer.

Purpose: generate synthetic launch-week trajectories calibrated to the two
real anchors we have (N=638 at 2026-11-19T02:30Z from the kit's practice
history; N~10000 at 2026-11-20T01:14Z from the live Exophase page, where the
observed rates are exactly consistent with integer counts over N=10000),
cut final-like cases (checkpoint 2026-11-21T11:30Z, target 2026-11-26T00:00Z),
and rank candidate forecasters by exact pinball loss on the logit scale.

Generative model
---------------
- Players tracked N(t): logistic growth, calibrated through the two anchors
  for a given saturation K.
- Each player joins N at time s (density ~ dN/dt) and earns trophy i after a
  delay D_i measured in accumulated *play-time* (weekend play intensity is
  higher), D_i ~ LogNormal(median m_i, sigma 0.8). A small reviewer seed
  joins before launch with pre-launch play time.
- E_i(t) = # players with s + D_i <= t (in play-time). rate = 100*E/N.
- Exophase page refreshes in batches: the observed read is the true state
  at the last refresh (every ~3h early, ~1h later); rates rounded to 2dp.
"""
import json
import math
import random
from datetime import datetime, timedelta, timezone

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Trophy delay medians (hours of accumulated play-time) by category.
# ids from the kit's practice case trophy list. Calibrated so the simulated
# world reproduces the live Exophase snapshot of 2026-11-20T01:14Z
# (N~10000, chapter 1 0.73%, Gone Fishing 99.99%): players average only a
# few play-hours a day, so clock-time progress is far slower than raw hours.
STORY = {3: 20.0, 4: 38.0, 5: 55.0, 6: 75.0, 7: 95.0}   # chapters 1-4, final
GRIND = {1: 600.0, 14: 450.0, 15: 260.0, 31: 160.0, 32: 125.0,
         35: 110.0, 19: 90.0, 30: 140.0, 54: 95.0, 25: 55.0}
EARLY = {2: 0.3, 20: 15.0, 8: 18.0, 16: 22.0, 21: 25.0, 44: 14.0,
         46: 16.0, 51: 20.0, 9: 40.0, 10: 36.0, 11: 44.0, 12: 42.0,
         13: 50.0, 17: 70.0, 18: 90.0, 22: 58.0, 23: 62.0, 24: 54.0,
         26: 60.0, 27: 64.0, 28: 70.0, 29: 48.0, 33: 24.0, 34: 28.0,
         36: 26.0, 37: 32.0, 38: 34.0, 39: 36.0, 40: 78.0, 41: 30.0,
         42: 95.0, 43: 72.0, 45: 22.0, 47: 45.0, 48: 48.0, 49: 40.0,
         50: 55.0, 52: 120.0, 53: 90.0, 55: 42.0}


def delay_median(tid):
    if tid in STORY:
        return STORY[tid]
    if tid in GRIND:
        return GRIND[tid]
    return EARLY.get(tid, 50.0)


def hours(t):
    return (t - LAUNCH).total_seconds() / 3600.0


def play_intensity(t, weekend_boost, play_frac=0.15):
    """Play-hours accumulated per clock hour. Players average a few
    play-hours a day (play_frac ~ 0.15), more on Sat/Sun and launch day."""
    wd = t.weekday()  # Mon=0; Nov 21 2026 is a Saturday
    base = play_frac
    if wd >= 5:
        base = play_frac * weekend_boost
    # launch-day surge in the first 24h
    h = hours(t)
    if 0 <= h < 24:
        base *= 1.4
    return base


def logistic_params(K, weekend_boost=1.0):
    """Solve logistic N(t)=K/(1+exp(-(h-h_mid)/tau)) through the two anchors.
    N=638 at h=2.5, N=10000 at h=25.23 (Nov 20 01:14Z)."""
    def u(n):
        return math.log(n / (K - n))
    h1, n1 = 2.5, 638.0
    h2, n2 = 25.23, 10000.0
    slope = (u(n2) - u(n1)) / (h2 - h1)  # per hour, in logit units
    tau = 1.0 / slope
    h_mid = h2 - u(n2) / slope
    return h_mid, tau


class World:
    """One simulated launch week."""

    def __init__(self, K=400000.0, delay_scale=1.0, weekend_boost=1.6,
                 refresh_hours=3.0, seed=0, reviewer_seed=25, play_frac=0.15,
                 delay_family="lognormal"):
        self.K = K
        self.delay_scale = delay_scale
        self.weekend_boost = weekend_boost
        self.refresh_hours = refresh_hours
        self.play_frac = play_frac
        self.delay_family = delay_family
        self.rng = random.Random(seed)
        self.h_mid, self.tau = logistic_params(K)
        self.tids = sorted(list(STORY) + list(GRIND) + list(EARLY))
        assert len(self.tids) == 55, len(self.tids)
        # play-time clock: cumulative integral of intensity from launch
        self._play_cache = {}

    def play_time(self, t):
        """Accumulated play-time hours from launch to t (numerical)."""
        h = hours(t)
        key = round(h, 2)
        if key in self._play_cache:
            return self._play_cache[key]
        # integrate intensity on a 0.25h grid
        total = 0.0
        step = 0.25
        n = int(h / step)
        for i in range(n):
            total += play_intensity(LAUNCH + timedelta(hours=i * step),
                                    self.weekend_boost,
                                    self.play_frac) * step
        total += play_intensity(LAUNCH + timedelta(hours=n * step),
                                self.weekend_boost,
                                self.play_frac) * (h - n * step)
        self._play_cache[key] = total
        return total

    def N(self, t):
        h = hours(t)
        return self.K / (1.0 + math.exp(-(h - self.h_mid) / self.tau))

    def _cohorts(self, t_end):
        """Cohort counts per join hour: regular players from dN/dt on an
        hourly grid, plus a small reviewer seed joining before launch."""
        h_end = hours(t_end)
        regular = {}
        prev = 0.0
        h = 0.0
        while h <= h_end:
            n = self.N(LAUNCH + timedelta(hours=h))
            dn = max(0.0, n - prev)
            if dn >= 0.5:
                regular[int(h)] = regular.get(int(h), 0) + int(round(dn))
            prev = n
            h += 1.0
        reviewers = {}
        for i in range(25):
            sh = -72 + i * 3  # spread over the 3 days before launch
            reviewers[sh] = reviewers.get(sh, 0) + 1
        return regular, reviewers

    def simulate(self, t_end):
        """Return dict: times (hourly grid), N(t), E_i(t) per trophy."""
        regular, reviewers = self._cohorts(t_end)
        h_end = hours(t_end)
        times = [LAUNCH + timedelta(hours=h) for h in range(0, int(h_end) + 1)]
        Ns = [self.N(t) for t in times]
        play = [self.play_time(t) for t in times]
        E = {tid: [0.0] * len(times) for tid in self.tids}
        # Precompute cohort (join play-time, count) pairs.
        cohorts = []
        for sh, cnt in sorted(regular.items()):
            cohorts.append((self.play_time(LAUNCH + timedelta(hours=sh)), cnt))
        for sh, cnt in sorted(reviewers.items()):
            cohorts.append((-float(sh), cnt))  # head start in play-hours
        for tid in self.tids:
            med = delay_median(tid) * self.delay_scale
            col = E[tid]
            for p0, cnt in cohorts:
                for ti, pt in enumerate(play):
                    avail = pt - p0
                    if avail <= 0:
                        continue
                    if self.delay_family == "weibull":
                        lam = med / (math.log(2.0) ** (1.0 / 1.5))
                        p = 1.0 - math.exp(-((avail / lam) ** 1.5))
                    else:
                        mu = math.log(med)
                        sig = 0.8
                        z = (math.log(avail) - mu) / sig
                        p = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
                    col[ti] += cnt * p
        return {"times": times, "N": Ns, "E": E, "play": play}

    def reads(self, sim, checkpoint, refresh_hours=None):
        """Cut house-style reads: 30-min cadence, observed state frozen at the
        last refresh. Returns list of read dicts like the kit's history."""
        rh = refresh_hours if refresh_hours is not None else self.refresh_hours
        times, Ns, E = sim["times"], sim["N"], sim["E"]
        out = []
        slot = datetime(2026, 11, 18, 23, 50, tzinfo=timezone.utc)
        last_refresh = None
        frozen = None
        while slot < checkpoint:
            # find true state at slot by interpolation on hourly grid
            h = hours(slot)
            idx = int(math.floor(h))
            frac = h - idx
            if idx < 0:
                n_true, e_true = 0.0, {tid: 0.0 for tid in self.tids}
            else:
                idx = min(idx, len(times) - 1)
                n_true = Ns[idx]
                e_true = {tid: E[tid][idx] for tid in self.tids}
                if idx + 1 < len(times) and frac > 0:
                    n_true = Ns[idx] * (1 - frac) + Ns[idx + 1] * frac
                    e_true = {tid: E[tid][idx] * (1 - frac) + E[tid][idx + 1] * frac
                              for tid in self.tids}
            if last_refresh is None or (slot - last_refresh).total_seconds() >= rh * 3600:
                last_refresh = slot
                frozen = (n_true, e_true)
            n_obs, e_obs = frozen
            rates = []
            for tid in self.tids:
                r = 100.0 * e_obs[tid] / n_obs if n_obs > 0 else 0.0
                rates.append(round(r + 1e-9, 2))
            out.append({
                "slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "at": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "players_tracked": int(round(n_obs)),
                "trophies_earned": int(round(sum(e_obs.values()))),
                "rates": rates,
            })
            slot += timedelta(minutes=30)
        return out


def pinball_loss(y, q, tau):
    """Exact scorer: 2*max(tau*d, (tau-1)*d), d = logit(y)-logit(q)."""
    def logit(x):
        x = min(99.99, max(0.01, x))
        return math.log((x / 100.0) / (1.0 - x / 100.0))
    d = logit(y) - logit(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_forecast(truth_rates, forecast_rows):
    """truth_rates: list of 55 true rates; forecast_rows: list of dicts
    {id, q:[5]} in trophy order. Returns mean pinball loss."""
    total, n = 0.0, 0
    for i, row in enumerate(forecast_rows):
        y = truth_rates[i]
        for tau, q in zip(LEVELS, row["q"]):
            total += pinball_loss(y, q, tau)
            n += 1
    return total / n


def load_practice_trophies():
    with open("/home/hatch/workspace/goals/make-fun-bounty-account-setup/hidden_files/kits/gta6-trophy-kit/gta6-trophy-rarity-forecast-01-kit/input/p00.in") as f:
        case = json.load(f)
    return case["trophies"]
