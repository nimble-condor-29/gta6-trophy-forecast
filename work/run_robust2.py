import sys
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast/work")
from run_backtest import run_backtest

# Weibull-delay worlds (different delay family than the forecaster assumes),
# plus lognormal worlds with different seeds for comparison.
scenarios = [
    {"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 2, "delay_family": "weibull"},
    {"K": 800000.0, "delay_scale": 1.3, "weekend_boost": 2.0, "refresh_hours": 2.0, "play_frac": 0.10, "seed": 2, "delay_family": "weibull"},
    {"K": 200000.0, "delay_scale": 0.8, "weekend_boost": 1.3, "refresh_hours": 6.0, "play_frac": 0.20, "seed": 2, "delay_family": "weibull"},
    {"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 2, "delay_family": "lognormal"},
    {"K": 800000.0, "delay_scale": 1.3, "weekend_boost": 2.0, "refresh_hours": 2.0, "play_frac": 0.10, "seed": 2, "delay_family": "lognormal"},
]
run_backtest(scenarios)
