import os
from dataclasses import dataclass, field
from scipy.optimize import minimize

import numpy as np
import pandas as pd
from loguru import logger
import datetime

from objective.objectives import Objective
from utils.saving_utils import save_df


@dataclass
class ScipyOptimizerConfig:
    method: str = 'L-BFGS-B'
    extra_args: dict = field(default_factory=lambda: {})


def scipy_optimizer(
        obj: Objective,
        config: ScipyOptimizerConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True
):
    results = []
    obj.iteration = 0
    initial_x = np.zeros(obj.dim)

    def logging_callback(x):
        loss = obj.loss(x, count_computation=False)
        grad_norm = np.linalg.norm(obj.grad(x, count_computation=False))
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
            iteration_type=f"outer",
            test_error=obj.error(x),
            train_error=obj.error(x, test_set=False),
            method_name="scipy_optimizer"
        ))

        obj.iteration += 1

        if obj.iteration % (max(1, iteration_budget // 100)) == 0:
            logger.info(f"running iteration {obj.iteration} of {config.method}  "
                        f"loss:{loss}, grad norm: {grad_norm}")

        if (obj.iteration) % 20 == 0 and save_results:
            outputs = get_outputs(results)
            os.makedirs(checkpoints_path, exist_ok=True)
            save_df(outputs, checkpoints_path)

    logging_callback(initial_x)

    options = dict(maxiter=iteration_budget, ftol=1e-12, gtol=1e-12, xtol=1e-12)
    options.update(config.extra_args)

    res = minimize(obj.loss, initial_x,
                   method=config.method, jac=obj.grad, hessp=lambda x, v: obj.hvp_factory(x)(v),
                   callback=logging_callback, options=options, tol=1e-15)

    return res.x, get_outputs(results)


def get_outputs(results):
    dfs = pd.DataFrame(results).rename(columns=dict(t="iteration"))
    return dfs
