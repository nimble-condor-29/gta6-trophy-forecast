import sys
sys.path.insert(0, "/home/hatch/workspace/makefun/gta6-trophy-forecast/work")
from run_backtest import run_backtest

scenarios = [
    {"K": 100000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 1},
    {"K": 2000000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 1},
    {"K": 400000.0, "delay_scale": 0.5, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 1},
    {"K": 400000.0, "delay_scale": 2.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 1},
    {"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 2.5, "refresh_hours": 2.0, "play_frac": 0.15, "seed": 1},
    {"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 6.0, "play_frac": 0.15, "seed": 1},
    {"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.08, "seed": 1},
    {"K": 400000.0, "delay_scale": 1.0, "weekend_boost": 1.6, "refresh_hours": 2.0, "play_frac": 0.30, "seed": 1},
]
run_backtest(scenarios)
