import os
from dataclasses import dataclass
from loguru import logger
import numpy as np

from algorithms.AGD import update_grad_method_results
from algorithms.ms_algorithm import get_outputs
from objective.objectives import Objective
from utils.saving_utils import save_df


@dataclass
class GDConfig:
    step_size: float = 4
    complexity_budget: int = 1e+10


def gradient_descent(
        obj: Objective,
        config: GDConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True
):
    results = []
    # initialize
    x = x_prev = initial_x = np.zeros(obj.dim)
    results = update_grad_method_results(x, initial_x, obj, results, "GD")

    while obj.iteration + 1 < iteration_budget:

        if obj.first_order_complexity() >= config.complexity_budget:
            break

        loss = obj.loss(x, count_computation=False)
        obj.iteration += 1

        x = x_prev - config.step_size * obj.grad(x_prev)
        x_prev = x

        results = update_grad_method_results(x, initial_x, obj, results, "GD")

        if (obj.iteration + 1) % 20 == 0:
            logger.info(f"||grad||: {np.linalg.norm(obj.grad(x_prev, count_computation=False))}, t={obj.iteration + 1}")
            logger.info(f"running iteration {obj.iteration + 1},  loss:{loss}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

    return x, get_outputs(results)


