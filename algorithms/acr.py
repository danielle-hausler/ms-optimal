import os
from dataclasses import dataclass
from typing import List

import numpy as np
from loguru import logger
import datetime

from algorithms.ms_algorithm import get_outputs, LAMBDA_FUNC, get_lambda_fn_args
from algorithms.solvers import cr_oracle
from objective.objectives import Objective
from utils.saving_utils import save_df


@dataclass
class ACRConfig:
    sec_ord_smoothness: float = None
    norm_x_opt: float = None
    mul_up: float = 2
    mul_down: float = 2
    lambda_func: str = "halving_doubling"
    first_lambda_guess: float = 0.1
    alpha: float = 1
    cr_oracle_accuracy: float = 1e-5
    lambda_newton: float = 1e-10


def acr_algorithm(
        obj: Objective,
        config: ACRConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True
):
    """Algorithm (4.8) in https://link.springer.com/content/pdf/10.1007/s10107-006-0089-x.pdf"""
    if config.sec_ord_smoothness is not None:
        if obj.sec_ord_smoothness is not None:
            obj.sec_ord_smoothness *= config.sec_ord_smoothness

        else:
            obj.sec_ord_smoothness = config.sec_ord_smoothness

    y = initial_x = np.zeros(obj.dim)
    lambda_prime = config.first_lambda_guess
    results = update_acr_results(obj, initial_x, initial_x, 0, lambda_prime, lambda_prime, [])

    # t = 1
    obj.iteration += 1

    obj.sec_ord_smoothness /= 2
    x, lambda_oracle_output, _, oracle_results = cr_oracle(obj, y, lambda_prime, config.mul_up, config.mul_down, config.cr_oracle_accuracy, config.lambda_newton)
    obj.sec_ord_smoothness *= 2
    sum_grad = 0
    v = initial_x
    y = 0.25 * x + 0.75 * v

    results.extend(oracle_results)
    results = update_acr_results(obj, x, initial_x, np.linalg.norm(x - y), lambda_prime, lambda_oracle_output, results)

    # t >= 2
    while obj.iteration + 1 < iteration_budget:
        obj.iteration += 1
        loss = obj.loss(x)

        t = obj.iteration
        lambda_fn_args = get_lambda_fn_args(config=config, obj=obj, lambda_prime_prev=lambda_prime,
                                            lambda_prev=lambda_oracle_output, prev_A=t/(t + 3))
        lambda_prime = LAMBDA_FUNC[config.lambda_func](**lambda_fn_args)

        x, lambda_oracle_output, _, oracle_results = cr_oracle(
            obj, y, lambda_prime, config.mul_up, config.mul_down,  config.cr_oracle_accuracy, config.lambda_newton)
        results.extend(oracle_results)
        results = update_acr_results(obj, x, initial_x, np.linalg.norm(x - y), lambda_prime, lambda_oracle_output, results)

        sum_grad += (t * (t + 1) / 2) * obj.grad(x)
        v = initial_x - np.sqrt(1 / (6 * obj.sec_ord_smoothness)) * sum_grad / np.linalg.norm(sum_grad) ** (1/2)  # equation before (4.10) in https://link.springer.com/content/pdf/10.1007/s10107-006-0089-x.pdf

        alpha = obj.iteration / (obj.iteration + 3)
        # alpha = 1   # force the method to become exactly Cubic Regularized Newton
        y = alpha * x + (1 - alpha) * v

        if (t + 1) % 20 == 0:
            logger.info(f"running iteration {obj.iteration + 1},  "
                        f"loss:{loss}")
            logger.info(f"||sum_grad||: {np.linalg.norm(sum_grad)}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

    return x, get_outputs(results)


def update_acr_results(obj: Objective, x: np.array, initial_x: np.array, r: float,
                            lambda_prime: float, lambda_oracle_output: float, results: List):
    results.append(dict(
        loss=obj.loss(x, count_computation=False),
        A=obj.iteration / (obj.iteration + 3),
        grad_norm=obj.norm(obj.grad(x, count_computation=False)),
        norm_x_x_0=obj.norm(x - initial_x),
        number_of_hessian_computations=obj.hessian_cnt,
        number_of_linear_system_solves=obj.linear_solves,
        number_of_hvp_computations=obj.hvp_cnt,
        number_of_grad_computations=obj.grad_cnt,
        number_of_fn_computations=obj.fn_cnt,
        lambda_prime=lambda_prime,
        lambda_oracle_output=lambda_oracle_output,
        t=obj.iteration,
        timestamp=datetime.datetime.now(),
        r=r,
        iteration_type="acr_outer_iteration",
        test_error=obj.error(x),
        train_error=obj.error(x, test_set=False),
        smoothness=obj.sec_ord_smoothness,
        method_name="ACR"
    ))

    return results
