import sys, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast/work")
import sim, forecast
from run_backtest import make_case, true_rates_at, CHECKPOINT, TARGET

def rescale(fc, factor):
    out = {"trophies": []}
    for row in fc["trophies"]:
        qs = row["q"]
        m = forecast.logit(qs[2])
        nq = []
        for q in qs:
            lg = m + (forecast.logit(q) - m) * factor
            nq.append(forecast.round4(forecast.clamp(forecast.inv_logit(lg))))
        for i in range(1, 5):
            if nq[i] < nq[i-1]:
                nq[i] = nq[i-1]
        out["trophies"].append({"id": row["id"], "q": nq})
    return out

trophies = sim.load_practice_trophies()
scenarios = []
for K in (200000.0, 800000.0):
    for ds in (0.8, 1.3):
        for pf in (0.10, 0.20):
            scenarios.append({"K": K, "delay_scale": ds, "weekend_boost": 1.6,
                              "refresh_hours": 2.0, "play_frac": pf, "seed": 0})
scenarios.append({"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6,
                  "refresh_hours": 2.0, "play_frac": 0.15, "seed": 2, "delay_family": "weibull"})
factors = [2.0, 2.5, 3.0, 4.0]
tot = {f: [] for f in factors}
cov = {f: [] for f in factors}
for sc in scenarios:
    w = sim.World(**sc)
    s = w.simulate(TARGET)
    truth, _ = true_rates_at(w, s, TARGET)
    case = make_case(w, s, trophies)
    base = forecast.cohort(case)
    for f in factors:
        fc = rescale(base, f)
        tot[f].append(sim.score_forecast(truth, fc["trophies"]))
        c = sum(1 for i, row in enumerate(fc["trophies"])
                if row["q"][0] <= truth[i] <= row["q"][4]) / 55.0
        cov[f].append(c)
for f in factors:
    v = tot[f]
    print(f"spread x{f}: mean loss {sum(v)/len(v):.4f} (min {min(v):.3f}, max {max(v):.3f})  90% cov {sum(cov[f])/len(cov[f]):.2f}")
