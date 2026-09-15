import math
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import utils
from registrators import CDRegistrator


def evaluate(cfg: DictConfig, value):
    """Run one full CD evaluation with the tuned hyperparameter set to ``value``."""
    trial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.update(trial, cfg.tuning.param, value)
    return CDRegistrator(trial).run()


def ternary_search(cfg: DictConfig, log):
    """Locate the minimum of the (unimodal) CD curve in log-scale.

    Implements Algorithm 1 of the paper: each iteration evaluates CD at two
    interior points and discards the third of the interval that cannot contain
    the minimum, so the search cost is logarithmic in the range rather than
    linear in the number of candidate values.
    """
    tuning = cfg.tuning
    key = "CD3" if cfg.metric == "both" else cfg.metric
    base = float(tuning.log_base)
    lo, hi = math.log(tuning.lower, base), math.log(tuning.upper, base)

    for iteration in range(1, tuning.iters + 1):
        if hi - lo <= tuning.tol:
            break
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        v1, v2 = int(base ** m1), int(base ** m2)
        r1, r2 = evaluate(cfg, v1), evaluate(cfg, v2)

        log(f"iter {iteration}/{tuning.iters}  {tuning.param}={v1}  " + fmt(r1, key))
        log(f"iter {iteration}/{tuning.iters}  {tuning.param}={v2}  " + fmt(r2, key))

        if r1[key] > r2[key]:
            lo = m1
        else:
            hi = m2

    best = int(base ** ((lo + hi) / 2))
    result = evaluate(cfg, best)
    log(f"selected  {tuning.param}={best}  " + fmt(result, key))
    return best, result


def fmt(result, key):
    fields = [f"{key}: {result[key]:.6f}"]
    if "TRE" in result:
        fields.append(f"TRE: {result['TRE']:.4f}")
    return "  ".join(fields)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Select a hyperparameter at testing time by minimising CD -- no labels used."""
    tag = cfg.datasets.get("case_idx", cfg.datasets.get("name", "pair"))
    path = Path(cfg.tuning.log_dir) / f"{cfg.datasets.type}_{tag}.log"
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(message):
        print(message)
        with path.open("a") as f:
            f.write(message + "\n")

    best, result = ternary_search(cfg, log)

    if cfg.out_csv:
        utils.append_csv(cfg.out_csv, {"stage": "tuned", cfg.tuning.param: best, **result})


if __name__ == "__main__":
    main()
