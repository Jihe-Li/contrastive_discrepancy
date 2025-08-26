import hydra
from omegaconf import DictConfig
from registrators import CDRegistrator


@hydra.main(version_base=None, config_path="configs", config_name="gaussian.yaml")
def main(cfg: DictConfig) -> None:

    registrator = CDRegistrator(cfg)
    CD2, CD3, tre_mean, tre_std = registrator.evaluate_cd()
    print(f"Case {cfg.case_idx}, CD2: {CD2:.4f}, CD3: {CD3:.4f}, TRE: {tre_mean[0]:.4f}±{tre_std[0]:.4f}")


if __name__ == "__main__":
    main()
