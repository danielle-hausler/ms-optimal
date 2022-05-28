import os
from dataclasses import dataclass
from typing import List
from scipy.optimize import brentq

import numpy as np
from loguru import logger
import datetime

from algorithms.ms_algorithm import get_outputs, LAMBDA_FUNC, get_lambda_fn_args, halving_doubling
from algorithms.solvers import cr_oracle
from objective.objectives import Objective
from utils.saving_utils import save_df


@dataclass
class AdaptiveACRConfig:
    sec_ord_smoothness: float = None
    mul_up: float = 2
    mul_down: float = 2
    first_lambda_guess: float = 0.1
    cr_oracle_accuracy: float = 1e-5
    lambda_newton: float = 1e-10
    drop_momentum: bool = False  # make this method unaccelerated


def adaptive_acr_algorithm(
        obj: Objective,
        config: AdaptiveACRConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True
):
    """Algorithm 4 in https://epubs.siam.org/doi/epdf/10.1137/19M1259432"""
    if config.sec_ord_smoothness is not None:
        if obj.sec_ord_smoothness is not None:
            obj.sec_ord_smoothness *= config.sec_ord_smoothness

        else:
            obj.sec_ord_smoothness = config.sec_ord_smoothness

    v = x = x_prev = initial_x = np.zeros(obj.dim)
    A = 0
    lambda_prime = config.first_lambda_guess
    results = []
    sum_grad = 0
    H_t = obj.sec_ord_smoothness
    results = update_adaptive_acr_results(obj, x, initial_x, 0, A, config.first_lambda_guess,
                                    config.first_lambda_guess, results)

    condition = lambda x, y, c: np.dot(obj.grad(x), y - x) >= 0.25 * (c ** (1 / 2)) * np.linalg.norm(obj.grad(x)) ** (3 / 2)

    while obj.iteration < iteration_budget:
        loss = obj.loss(x)
        done = False
        i = 0
        lambda_query = lambda_prime
        while not done:
            logger.log("TRACE", f"running inner adaptive_acr iteration {i}")
            c = 1 / (2 ** i * H_t)
            c_a = c / 2 ** 5
            H = 2 ** i * H_t
            obj.sec_ord_smoothness = (3 / 2) * H
            if obj.iteration == 0:
                a = c_a
            else:
                a = brentq(lambda a: a ** 3 - c_a * (A + a) ** 2, 0, max(A, 4 * c_a))
            gamma = (a / (A + a))
            y = (1-gamma) * x_prev + gamma * v
            x, lambda_oracle_output, _, oracle_results = cr_oracle(obj, y, lambda_query, config.mul_up, config.mul_down,
                                                                config.cr_oracle_accuracy, config.lambda_newton)
            results.extend([{**d, 'i': i} for d in oracle_results])
            results = update_adaptive_acr_results(obj, x, initial_x, np.linalg.norm(x - y), A, lambda_prime, lambda_oracle_output,
                                            results, iteration_type="inner", i=i)

            lambda_query = halving_doubling(lambda_oracle_output, lambda_query)
            if condition(x, y, c):
                done = True
            else:
                i += 1
        results = update_adaptive_acr_results(obj, x, initial_x, np.linalg.norm(x - y), A, lambda_prime,
                                        lambda_oracle_output, results)
        obj.iteration += 1
        x_prev = x
        A += a
        H_t = H_t * 2**(i-1)
        lambda_prime = lambda_query / 2
        sum_grad += a * obj.grad(x)
        v = initial_x - (sum_grad / np.linalg.norm(sum_grad)**(1/2))

        if (obj.iteration + 1) % 20:
            logger.info(f"running iteration {obj.iteration} "
                        f"loss:{loss}")
            logger.info(f"sum_grad: {np.linalg.norm(sum_grad)}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

    return x, get_outputs(results)


def update_adaptive_acr_results(obj: Objective, x: np.array, initial_x: np.array, r: float, A: float, lambda_prime: float,
                          lambda_oracle_output: float, results: List, iteration_type="outer", i=np.nan):
    results.append(dict(
        loss=obj.loss(x, count_computation=False),
        A=A,
        grad_norm=obj.norm(obj.grad(x, count_computation=False)),
        norm_x_x_0=obj.norm(x - initial_x),
        number_of_hessian_computations=obj.hessian_cnt,
        number_of_linear_system_solves=obj.linear_solves,
        number_of_hvp_computations=obj.hvp_cnt,
        number_of_grad_computations=obj.grad_cnt,
        number_of_fn_computations=obj.fn_cnt,
        lambda_oracle_output=lambda_oracle_output,
        lambda_prime=lambda_prime,
        t=obj.iteration,
        timestamp=datetime.datetime.now(),
        r=r,
        i=i,
        iteration_type=f"adaptive_acr_{iteration_type}_iteration",
        test_error=obj.error(x),
        train_error=obj.error(x, test_set=False),
        method_name="adaptive_ACR"
    ))

    return results

