import os
from dataclasses import dataclass
from datetime import datetime
from typing import List
from loguru import logger
import numpy as np

from algorithms.ms_algorithm import get_outputs
from objective.objectives import Objective
from utils.saving_utils import save_df


@dataclass
class AGDConfig:
    step_size: float = 4
    complexity_budget: int = 1e+10


def AGD(
        obj: Objective,
        config: AGDConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True
):
    """
    Algorithm from https://vsokolov.org/courses/750/2018/files/acr.pdf'
    """
    results = []
    x = x_prev = initial_x = np.zeros(obj.dim)
    results = update_grad_method_results(x, initial_x, obj, results,  "AGD")
    # initialize
    t = t_prev = 1
    y = x

    while obj.iteration + 1 < iteration_budget:

        if obj.first_order_complexity() >= config.complexity_budget:
            break

        loss = obj.loss(x, count_computation=False)
        obj.iteration += 1

        x = y - config.step_size * obj.grad(y)
        t = 0.5 + 0.5 * np.sqrt(1 + 4 * (t_prev ** 2))
        y = x + (t_prev - 1) / t * (x-x_prev)

        t_prev = t
        x_prev = x

        results = update_grad_method_results(x, initial_x, obj, results, "AGD")

        if (t + 1) % 20 == 0:
            logger.info(f"||grad||: {np.linalg.norm(obj.grad(y, count_computation=False))}, t={t}")
            logger.info(f"running iteration {obj.iteration + 1},  "
                        f"loss:{loss}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

    return x, get_outputs(results)


def update_grad_method_results(x: np.array, initial_x: np.array, obj: Objective, results: List, method_name: str):
    loss = obj.loss(x, count_computation=False)
    grad_norm = np.linalg.norm(obj.grad(x, count_computation=False))
    results.append(dict(
            loss=loss,
            A=np.nan,
            grad_norm=grad_norm,
            norm_x_x_0=obj.norm(x - initial_x),
            number_of_hessian_computations=obj.hessian_cnt,
            number_of_linear_system_solves=obj.linear_solves,
            number_of_hvp_computations=obj.hvp_cnt,
            number_of_grad_computations=obj.grad_cnt,
            number_of_fn_computations=obj.fn_cnt,
            t=obj.iteration,
            timestamp=datetime.now(),
            iteration_type=f"outer",
            test_error=obj.error(x),
            train_error=obj.error(x, test_set=False),
            method_name=method_name
        ))
    return results