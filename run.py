import hydra
from omegaconf import DictConfig

import utils
from registrators import CDRegistrator


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Evaluate Contrastive Discrepancy for one model configuration."""
    registrator = CDRegistrator(cfg)
    result = registrator.run()
    print(registrator.summary(result))

    if cfg.out_csv:
        row = {"dataset": registrator.dataset.name,
               cfg.tuning.param: utils.get_by_path(cfg, cfg.tuning.param),
               **result}
        utils.append_csv(cfg.out_csv, row)


if __name__ == "__main__":
    main()
