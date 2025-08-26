import os
import numpy as np
from omegaconf import DictConfig
from hydra import initialize, compose

from registrators import CDRegistrator


def evaluate_model_cd(cfg, param):
    cfg.network.max_densify_num = param
    
    registrator = CDRegistrator(cfg)
    _, CD3, tre_mean, _ = registrator.evaluate_cd()

    return CD3, tre_mean


def ternary_search_log_scale(case_idx, lambda_l, lambda_r, log_base, iters=5):
    """
    Perform ternary search in log scale to find optimal lambda.
    Input:
        log_l: log10(lambda_1)
        log_r: log10(lambda_2)
        eps:   tolerance for convergence (in log10 scale)
    Output:
        best_lambda: optimal lambda value (not in log scale)
        best_metric: minimal metric value
    """
    with initialize(version_base=None, config_path="configs"):
        cfg: DictConfig = compose(config_name="gaussian.yaml")

    cfg.case_idx = case_idx

    log_l = eval('np.log%d' % log_base)(lambda_l)
    log_r = eval('np.log%d' % log_base)(lambda_r)

    for i in range(iters):
        m1 = log_l + (log_r - log_l) / 3
        m2 = log_r - (log_r - log_l) / 3
        f1, tre1 = evaluate_model_cd(cfg, int(log_base ** m1))
        f2, tre2 = evaluate_model_cd(cfg, int(log_base ** m2))
        if f1 > f2:
            log_l = m1
        else:
            log_r = m2
        
        message = f"Case {cfg.case_idx}\n" 
        message += f"Iteration {i + 1}/{iters}; Param: {int(log_base ** m1)}, CD3: {f1.detach().cpu().numpy():.6f}, TRE: {tre1[0]:.6f}\n"
        message += f"Iteration {i + 1}/{iters}; Param: {int(log_base ** m2)}, CD3: {f2.detach().cpu().numpy():.6f}, TRE: {tre2[0]:.6f}\n"

        path = f'outputs/DIRLab_tuning_iter{iters}_CD/log_{cfg.case_idx}.txt'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            f.write(message)

    best_log_lambda = (log_l + log_r) / 2
    best_lambda = int(log_base ** best_log_lambda)
    best_metric = eval('evaluate_model_%s' % 'CD')(cfg, best_lambda)
    message = f"Final best param: {best_lambda}, Metric: {best_metric[0].detach().cpu().numpy():.6f}, TRE: {best_metric[1][0]:.6f}\n"
    with open(path, 'a') as f:
        f.write(message)
    return best_lambda, best_metric


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_l', default=200, type=int)
    parser.add_argument('--lambda_r', default=204800, type=int)
    parser.add_argument('--log_base', default=2, type=int)
    parser.add_argument('--iters', default=5, type=int)
    args = parser.parse_args()

    # Run ternary search
    for case_idx in range(1, 11):
        best_lambda, best_metric = ternary_search_log_scale(case_idx,
                                                            args.lambda_l, 
                                                            args.lambda_r, 
                                                            args.log_base, 
                                                            args.iters)

        print(f"Case_idx{case_idx}, Best lambda: {best_lambda:.6f}")
        print(f"Case_idx{case_idx}, Metric at best lambda: {best_metric[0].detach().cpu().numpy():.6f}")
        print(f"Case_idx{case_idx}, TRE at best lambda: {best_metric[1][0]:.6f}")
