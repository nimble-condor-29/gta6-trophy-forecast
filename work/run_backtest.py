"""Backtest runner: rank candidate forecasters across simulated worlds."""
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast/work")
import sim
import forecast

CHECKPOINT = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)
LIVE_SNAP = datetime(2026, 11, 20, 1, 14, tzinfo=timezone.utc)

# Live Exophase snapshot 2026-11-20T01:14Z (trophy id -> rate %).
LIVE = {1: 0.03, 2: 99.99, 3: 0.73, 4: 0.14, 5: 0.06, 6: 0.06, 7: 0.06,
        8: 0.24, 9: 0.06, 10: 0.06, 11: 0.06, 12: 0.06, 13: 0.06, 14: 0.03,
        15: 0.03, 16: 0.04, 17: 0.04, 18: 0.03, 19: 0.04, 20: 1.89,
        21: 0.41, 22: 0.04, 23: 0.04, 24: 0.04, 25: 0.03, 26: 0.03,
        27: 0.04, 28: 0.03, 29: 0.04, 30: 0.03, 31: 0.03, 32: 0.03,
        33: 0.09, 34: 0.05, 35: 0.24, 36: 0.06, 37: 0.14, 38: 0.14,
        39: 0.06, 40: 0.04, 41: 0.04, 42: 0.04, 43: 0.04, 44: 0.08,
        45: 0.04, 46: 0.09, 47: 0.04, 48: 0.05, 49: 0.04, 50: 0.11,
        51: 0.06, 52: 0.04, 53: 0.05, 54: 0.04, 55: 0.04}


def true_rates_at(world, s, t):
    """Interpolated true rates (2dp) at time t from a simulation."""
    times, Ns, E = s["times"], s["N"], s["E"]
    h = sim.hours(t)
    idx = int(h)
    if idx >= len(times) - 1:
        idx = len(times) - 2
        frac = 1.0
    else:
        frac = h - idx
    n = Ns[idx] * (1 - frac) + Ns[idx + 1] * frac
    rates = []
    for tid in world.tids:
        e = E[tid][idx] * (1 - frac) + E[tid][idx + 1] * frac
        rates.append(round(100.0 * e / n, 2) if n > 0 else 0.0)
    return rates, n


def make_case(world, s, trophies):
    hist = world.reads(s, CHECKPOINT)
    horizon = int((TARGET - CHECKPOINT).total_seconds() / 60)
    return {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": "2026-11-21T11:30:00Z",
        "horizon_minutes": horizon,
        "target": "2026-11-26T00:00:00Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": hist,
        "notes": "backtest",
    }


def sanity_check():
    """Compare one world against the live Nov 20 snapshot."""
    trophies = sim.load_practice_trophies()
    w = sim.World(K=400000.0, delay_scale=1.0, weekend_boost=1.6,
                  refresh_hours=3.0, seed=0)
    s = w.simulate(TARGET)
    rates, n = true_rates_at(w, s, LIVE_SNAP)
    print(f"sanity: simulated N at live snapshot = {n:,.0f} (observed ~10000)")
    print(f"sanity: last history N at checkpoint = "
          f"{s['N'][int(sim.hours(CHECKPOINT))]:,.0f}")
    import math
    # print a few key trophies
    for tid in (2, 3, 4, 5, 7, 8, 20, 21, 33, 35, 1, 14):
        print(f"  trophy {tid:2d}: sim {rates[tid-1]:6.2f}%  live {LIVE[tid]:6.2f}%")
    # mean abs logit error across trophies
    tot = 0.0
    for tid, obs in LIVE.items():
        tot += abs(math.log(max(rates[tid - 1], 0.01) / 100.0 /
                            (1 - max(rates[tid - 1], 0.01) / 100.0))
                   - math.log(max(obs, 0.01) / 100.0 /
                              (1 - max(obs, 0.01) / 100.0)))
    print(f"  mean |logit err| vs live snapshot: {tot / 55:.3f}")


def run_backtest(scenarios):
    trophies = sim.load_practice_trophies()
    methods = {
        "persist": forecast.persist,
        "curve": forecast.curve,
        "hazard": forecast.hazard,
        "hazard_chain": forecast.hazard_chain,
        "cohort": forecast.cohort,
        "blend2_70": lambda c: forecast.blend2(c, 0.7, 0.3),
        "blend2_50": lambda c: forecast.blend2(c, 0.5, 0.5),
        "blend85": lambda c: forecast.blend(c, 0.85, 0.15, 0.0),
    }
    totals = {k: [] for k in methods}
    for sc in scenarios:
        t0 = time.time()
        w = sim.World(**sc)
        s = w.simulate(TARGET)
        truth, n_tgt = true_rates_at(w, s, TARGET)
        case = make_case(w, s, trophies)
        n_ck = s["N"][int(sim.hours(CHECKPOINT))]
        line = (f"K={sc['K'] / 1000:.0f}k ds={sc['delay_scale']} "
                f"wb={sc['weekend_boost']} | N_ck={n_ck:,.0f} "
                f"N_tgt={n_tgt:,.0f} | ")
        for name, fn in methods.items():
            fc = fn(case)
            loss = sim.score_forecast(truth, fc["trophies"])
            totals[name].append(loss)
            line += f"{name}={loss:.3f} "
        print(line + f"({time.time() - t0:.0f}s)")
    print("\n=== mean loss across scenarios ===")
    for name, vals in sorted(totals.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        print(f"  {name:12s} {sum(vals) / len(vals):.4f}  "
              f"(min {min(vals):.3f}, max {max(vals):.3f})")


if __name__ == "__main__":
    sanity_check()
    print()
    scenarios = []
    for K in (200000.0, 800000.0):
        for ds in (0.8, 1.3):
            for pf in (0.10, 0.20):
                scenarios.append({"K": K, "delay_scale": ds,
                                  "weekend_boost": 1.6, "refresh_hours": 2.0,
                                  "play_frac": pf, "seed": 0})
    run_backtest(scenarios)
