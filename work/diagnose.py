import sys, math
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast/work")
import sim, forecast
from run_backtest import make_case, true_rates_at, CHECKPOINT, TARGET

trophies = sim.load_practice_trophies()
for sc in ({"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6,
            "refresh_hours": 2.0, "play_frac": 0.15, "seed": 0},
           {"K": 800000.0, "delay_scale": 1.3, "weekend_boost": 1.6,
            "refresh_hours": 2.0, "play_frac": 0.10, "seed": 0}):
    w = sim.World(**sc)
    s = w.simulate(TARGET)
    truth, n_tgt = true_rates_at(w, s, TARGET)
    case = make_case(w, s, trophies)
    # N fit check
    _, times, Ns, _, target_h, _ = forecast.case_arrays(case)
    nfun = forecast.fit_logistic_N(times, Ns)
    print(f"K={sc['K']/1000:.0f}k ds={sc['delay_scale']} pf={sc['play_frac']}: "
          f"N_last={Ns[-1]:,.0f} Nfit_tgt={nfun(target_h):,.0f} Ntrue_tgt={n_tgt:,.0f}")
    fc = forecast.hazard_chain(case)["trophies"]
    errs = []
    for i, row in enumerate(fc):
        errs.append(forecast.logit(truth[i]) - forecast.logit(row["q"][2]))
    errs_sorted = sorted(errs)
    import statistics
    print(f"  median logit err: mean {sum(errs)/len(errs):+.2f} "
          f"rms {math.sqrt(sum(e*e for e in errs)/len(errs)):.2f} "
          f"p10 {errs_sorted[5]:+.2f} p90 {errs_sorted[49]:+.2f}")
    for tid in (3, 4, 7, 2, 20, 21, 1):
        i = tid - 1
        print(f"    trophy {tid:2d}: pred_med {fc[i]['q'][2]:7.2f}% "
              f"truth {truth[i]:7.2f}%  last {case['history'][-1]['rates'][i]:7.2f}%")
