import sys, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast/work")
import sim, forecast
from run_backtest import make_case, true_rates_at, CHECKPOINT, TARGET

def n_power(times, Ns, target_h, window=48.0):
    t_end = times[-1]
    xs, ys = [], []
    for t, n in zip(times, Ns):
        if t >= t_end - window and n > 0:
            xs.append(math.log1p(t)); ys.append(math.log(n))
    if len(xs) < 3: return Ns[-1]
    a, b, _ = forecast.ols(xs, ys)
    return math.exp(a + b * math.log1p(target_h))

def n_damped_exp(times, Ns, target_h, window=24.0, damp=0.5):
    t_end = times[-1]
    pts = [(t, n) for t, n in zip(times, Ns) if t >= t_end - window and n > 0]
    if len(pts) < 2: return Ns[-1]
    g = (math.log(pts[-1][1]) - math.log(pts[0][1])) / max(1.0, pts[-1][0] - pts[0][0])
    return Ns[-1] * math.exp(g * damp * (target_h - t_end))

def n_const(times, Ns, target_h):
    return Ns[-1]

trophies = sim.load_practice_trophies()
scenarios = []
for K in (200000.0, 400000.0, 800000.0, 2000000.0):
    for ds in (0.8, 1.3):
        for pf in (0.10, 0.20):
            scenarios.append({"K": K, "delay_scale": ds, "weekend_boost": 1.6,
                              "refresh_hours": 2.0, "play_frac": pf, "seed": 0})
methods = {"logistic": None, "power48": n_power, "damped_exp": n_damped_exp, "const": n_const}
errs = {k: [] for k in methods}
for sc in scenarios:
    w = sim.World(**sc)
    s = w.simulate(TARGET)
    _, n_tgt = true_rates_at(w, s, TARGET)
    case = make_case(w, s, trophies)
    _, times, Ns, _, target_h, _ = forecast.case_arrays(case)
    for name, fn in methods.items():
        pred = forecast.fit_logistic_N(times, Ns)(target_h) if fn is None else fn(times, Ns, target_h)
        errs[name].append(math.log(pred / n_tgt))
    print(f"K={sc['K']/1000:.0f}k ds={sc['delay_scale']} pf={sc['play_frac']}: "
          f"Ntrue={n_tgt:,.0f} " +
          " ".join(f"{k}={v[-1]:+.2f}" for k, v in errs.items()))
print("\nmean |log err| of N_tgt:")
for k, v in errs.items():
    print(f"  {k:10s} {sum(abs(x) for x in v)/len(v):.3f}  bias {sum(v)/len(v):+.3f}")
