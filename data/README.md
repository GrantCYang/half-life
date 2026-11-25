# Aggregate data

`forecast/` contains seven published tables: calibrated Brier, frozen baseline references, paired adaptation gains, outcome strata, deterministic and interpolated fixed-rate recall, and paired recall differences. The figure script reads these tables without fitting new models or recalibrating scores.

`caption/` contains the recorded downstream caption aggregate results. `historical/` contains earlier-implementation aggregates; these are not the current repaired forecaster's main results. They are kept separate to prevent mixing experiments.

The Brier and recall figures regenerate from `forecast/`. Other paper figures are supplied as finished publication illustrations. Per-row records and the internal statistical replay packages are omitted. Original aggregate tables are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), with attribution to Yixuan Yang and the accompanying paper. Third-party material is excluded from this license grant.
