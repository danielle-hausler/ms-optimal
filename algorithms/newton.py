import os
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from loguru import logger
import datetime

from objective.objectives import Objective
from utils.saving_utils import save_df


@dataclass
class NewtonMethodConfig:
    lambda_newton: float = 1e-10
    exact_line_search: bool = False
    exact_line_search_bound: float = 7.0


def newton_method(
        obj: Objective,
        config: NewtonMethodConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True
):
    """Newton's mehod with very exact line search"""
    initial_x = x = np.zeros(obj.dim)
    results = []
    obj.iteration = 0

    while obj.iteration + 1 < iteration_budget:
        obj.iteration += 1

        grad_x = obj.grad(x)
        hess_x = obj.hessian(x)
        p = obj.solve_linear_system(config.lambda_newton, hess_x, -grad_x)
        if config.exact_line_search:
            res_ = minimize_scalar(lambda alpha: obj.loss(x + (2 ** alpha) * p, count_computation=True),
                                   bounds=[-config.exact_line_search_bound, config.exact_line_search_bound], method='bounded')
            coef = 2 ** res_.x
        else:
            coef = 1.0
        x = x + coef * p
        loss = obj.loss(x, count_computation=False)
        grad_norm = np.linalg.norm(grad_x)
        update_newton_results(obj, loss, grad_norm, x, initial_x, coef, p, results)

        if (obj.iteration + 1) % 20 == 0:
            logger.info(f"running iteration {obj.iteration} of Newton method  "
                        f"loss:{loss}, grad norm: {grad_norm}, LS coef = {coef}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

    return x, get_outputs(results)


def update_newton_results(obj: Objective, loss: float, grad_norm: float,
                          x: np.array, initial_x: np.array, coef: float, p: np.array,
                          results: List):
    results.append(dict(
        loss=loss,
        grad_norm=grad_norm,
        norm_x_x_0=obj.norm(x - initial_x),
        number_of_hessian_computations=obj.hessian_cnt,
        number_of_linear_system_solves=obj.linear_solves,
        number_of_hvp_computations=obj.hvp_cnt,
        number_of_grad_computations=obj.grad_cnt,
        number_of_fn_computations=obj.fn_cnt,
        t=obj.iteration,
        timestamp=datetime.datetime.now(),
        ls_coef=coef,
        r=coef * np.linalg.norm(p),
        iteration_type="newton_outer_iteration",
        test_error=obj.error(x),
        train_error=obj.error(x, test_set=False),
        method_name="newton"
    ))

    return results


def get_outputs(results):
    dfs = pd.DataFrame(results).rename(columns=dict(t="iteration"))
    return dfs
