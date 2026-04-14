# UltraNest Live Monitor

Standalone dashboard for watching UltraNest progress while ASAP retrievals are running.

## What It Shows

- Evidence trajectory (`logz`) from `debug.log`
- Remaining evidence fraction (`remainder_fraction`)
- Likelihood bounds (`Lmin` and `Lmax`)
- Throughput (`ncalls / second`)
- B-distribution snapshot for filling factors with 68% errors (uses `results/points.hdf5` live fallback, then `chains/weighted_post.txt` when available)
- Phase 2 posterior evolution: selectable parameter median and 68% interval trajectory from live/final posterior checkpoints
- Text summary of latest filling-factor posteriors and uncertainties
- Live summary including latest iteration, call count, and `results.json` fields

Filling-factor handling in Phase 2:
- The dashboard prioritizes magnetic filling factors in the posterior panels.
- The `0 kG` filling factor is derived as remainder: `a0 = 1 - sum(a_i)` from the fitted free components.
- Higher-kG free factors are kept (not truncated) before non-field parameters are added.

## Install

```bash
cd /net/vdesk/data2/cobelens/MRP/new/code_vibing/ultranest_live_monitor
PYTHONUSERBASE=/net/vdesk/data2/cobelens/.local \
PIP_CACHE_DIR=/net/vdesk/data2/cobelens/.cache/pip \
pip install --user -r requirements.txt
```

## Run

```bash
PYTHONUSERBASE=/net/vdesk/data2/cobelens/.local python app.py \
  --log-dir /net/vdesk/data2/cobelens/MRP/new/asap-nested-sampling-integration/testing/nested_sampling/gl_411/output_gl_411_8kG/ultranest_logdir \
  --port 8062
```

Then open `http://127.0.0.1:8062`.

## Notes

- The monitor is read-only and does not modify any run files.
- It is safe to start before `info/results.json` or `chains/` files exist.
- During active writes, the parser retries on partial JSON/text and updates at the next poll.
- During active sampling, posterior panels are computed from `results/points.hdf5` as a live approximation; final `weighted_post.txt` takes over automatically when written.
- Phase 2 posterior evolution is automatically downsampled to checkpoint snapshots to stay responsive on large runs.
