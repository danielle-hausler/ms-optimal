import os
from dataclasses import dataclass
from functools import partial
from typing import List, Literal
import numpy as np
import inspect
import datetime

import pandas as pd
from numpy.linalg import norm
from loguru import logger

from algorithms.bisection import ms_bisection
from objective.objectives import Objective
from algorithms.solvers import cr_oracle, amsn_fo_oracle, amsn_oracle
from utils.saving_utils import save_experiment_outputs, save_df


@dataclass
class MSConfig:
    lambda_func: str = "halving_doubling"
    alpha: float = 2
    norm_x_opt: float = None
    first_lambda_guess: float = 0.1
    drop_momentum: float = False
    stop_loss: float = 1e+5
    min_grad_norm: float = 1e-15
    exact_first_bisection: bool = True  # indicates if we should take the first oracle's output as lambda_prime
    first_order_complexity_budget: int = 1e+10  # complexity budget for first order computations
    oracle_type: str = "amsn"  # currently supported: amsn, amsn_fo_oracle and cr_oracle
    # configs for ms_oracle
    sigma: float = 1 / 2
    lazy_oracle: bool = False
    # configs for cr oracle
    sec_ord_smoothness: float = None
    cr_oracle_accuracy: float = 1e-5
    # configs for both type of oracles
    mul_up: float = 2
    mul_down: float = 2
    lambda_newton: float = 1e-10  # indicates a lambda value so low it's effectively a Newton step


@dataclass
class MSBisectionConfig(MSConfig):
    lower_rho: float = 1
    upper_rho: float = 4.001
    max_bisection_iterations: int = 1000


def ms_algorithm(
        obj: Objective,
        config: MSBisectionConfig,
        iteration_budget: int,
        checkpoints_path: str,
        save_results: bool = True

):
    if config.sec_ord_smoothness is not None:
        if obj.sec_ord_smoothness is not None:
            obj.sec_ord_smoothness *= config.sec_ord_smoothness

        else:
            obj.sec_ord_smoothness = config.sec_ord_smoothness

    oracle = get_oracle_fn(config)

    bisection_kwargs = dict(
        oracle=oracle,
        mul_up=config.mul_up,
        mul_down=config.mul_down,
        lower_rho=config.lower_rho,
        upper_rho=config.upper_rho,
        max_iterations=config.max_bisection_iterations
    )

    x = initial_x = v = np.zeros(obj.dim)
    A = 0

    results = init_results_list(obj, x, config.first_lambda_guess, method_name=ms_algorithm_method_name(config))
    results[0]['lambda_bisection_output'] = config.first_lambda_guess
    lambda_fn_args = get_lambda_fn_args(config=config, obj=obj, prev_A=A)

    while obj.iteration + 1 < iteration_budget:
        total_budget = obj.fn_cnt + obj.grad_cnt + obj.hvp_cnt

        if config.oracle_type in ["amsn_fo_oracle",
                                  "gradient_step_ms_oracle"] and total_budget >= config.first_order_complexity_budget:
            break

        lambda_prime = LAMBDA_FUNC[config.lambda_func](**lambda_fn_args)

        obj.iteration += 1

        # do not use lazy oracle in the first iteration
        if obj.iteration == 1 and "lazy" in inspect.signature(oracle).parameters:
            kwargs = {**bisection_kwargs, 'oracle': partial(oracle, lazy=False)}
        else:
            kwargs = bisection_kwargs
        x_next, lambda_oracle, lambda_bisection, grad, bisection_results = ms_bisection(obj, x, v, A, lambda_prime,
                                                                                        **kwargs)

        results.extend(bisection_results)

        a_prime = (1 / (2 * lambda_bisection)) * (1 + np.sqrt(1 + 4 * A * lambda_bisection))
        A_next = A + a_prime
        y = (A / A_next) * x + (a_prime / A_next) * v
        v_next = v - a_prime * grad

        x, v, A = x_next, v_next, A_next

        loss = obj.loss(x, count_computation=False)

        if (obj.iteration + 1) % 20 == 0:
            logger.info(f"first order complexity = {total_budget}")
            logger.info(f"running iteration {obj.iteration + 1}  "
                        f"loss:{loss}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

        lambda_fn_args = get_lambda_fn_args(config=config, obj=obj, prev_A=A, lambda_prime_prev=lambda_prime,
                                            lambda_prev=lambda_bisection)

        grad_norm = norm(obj.grad(x, count_computation=False))

        results.append(dict(
            loss=loss,
            A=A,
            lambda_prime=lambda_prime,
            lambda_bisection_output=lambda_bisection,
            lambda_oracle_output=lambda_oracle,
            grad_norm=grad_norm,
            norm_x_x_0=norm(x - initial_x),
            number_of_hessian_computations=obj.hessian_cnt,
            number_of_linear_system_solves=obj.linear_solves,
            number_of_hvp_computations=obj.hvp_cnt,
            number_of_grad_computations=obj.grad_cnt,
            number_of_fn_computations=obj.fn_cnt,
            r=np.linalg.norm(x - y),
            t=obj.iteration,
            timestamp=datetime.datetime.now(),
            iteration_type="ms_outer_iteration",
            test_error=obj.error(x),
            train_error=obj.error(x, test_set=False),
            method_name=ms_algorithm_method_name(config)
        ))

        if np.nan_to_num(loss) > config.stop_loss:
            break

        if grad_norm <= config.min_grad_norm:
            break

    return x, get_outputs(results)


@dataclass
class MSNoBisectionConfig(MSConfig):
    A_prime: bool = True
    best_x_update: bool = False


def opt_ms_algorithm(
        obj: Objective,
        config: MSNoBisectionConfig,
        checkpoints_path: str,
        iteration_budget: int,
        save_results: bool = True

):
    idx = None
    x = initial_x = v = np.zeros(obj.dim)
    A = 0
    oracle = get_oracle_fn(config)

    results = init_results_list(obj, x, config.first_lambda_guess, method_name=opt_ms_algorithm_method_name(config))
    lambda_fn_args = get_lambda_fn_args(config=config, obj=obj, prev_A=A)

    if config.sec_ord_smoothness is not None:
        if obj.sec_ord_smoothness is not None:
            obj.sec_ord_smoothness *= config.sec_ord_smoothness
        else:
            obj.sec_ord_smoothness = config.sec_ord_smoothness

    while obj.iteration + 1 < iteration_budget:
        total_budget = obj.fn_cnt + obj.grad_cnt + obj.hvp_cnt

        if config.oracle_type in ["amsn_fo_oracle",
                                  "gradient_step_ms_oracle"] and total_budget >= config.first_order_complexity_budget:
            break
        lambda_prime = LAMBDA_FUNC[config.lambda_func](**lambda_fn_args)

        obj.iteration += 1

        a_prime = (1 / (2 * lambda_prime)) * (1 + np.sqrt(1 + 4 * A * lambda_prime))
        A_next = A + a_prime
        if config.drop_momentum:
            y = x
        else:
            y = (A / A_next) * x + (a_prime / A_next) * v

        # do not use lazy oracle in the first iteration
        if obj.iteration == 1 and "lazy" in inspect.signature(oracle).parameters:
            w_next, lambda_oracle, grad, oracle_results = oracle(obj=obj, y=y, lambda_query=lambda_prime, lazy=False)
        else:
            w_next, lambda_oracle, grad, oracle_results = oracle(obj=obj, y=y, lambda_query=lambda_prime)
        results.extend(oracle_results)

        if obj.iteration == 1 and config.exact_first_bisection:
            lambda_prime = lambda_oracle
            a_prime = (1 / (2 * lambda_prime)) * (1 + np.sqrt(1 + 4 * A * lambda_prime))
            A_next = A + a_prime

        if config.A_prime:
            A_prime = A_next
            gamma = min(1, lambda_prime / lambda_oracle)
            a = gamma * a_prime
            A_next = A + a
            v_next = v - a * grad
            if config.best_x_update or config.drop_momentum:
                x_candidates = [x, w_next]
                idx = np.argmin([obj.loss(x, count_computation=False) for x in x_candidates])
                x_next = x_candidates[idx]
            else:
                x_next = (1 - gamma) * A * x / A_next + gamma * A_prime * w_next / A_next

        else:
            v_next = v - a_prime * grad
            x_next = w_next

        x, v, A = x_next, v_next, A_next

        loss = obj.loss(x, count_computation=False)

        if (obj.iteration + 1) % 20 == 0:
            logger.info(f"first order complexity = {total_budget}")
            logger.info(f"running iteration {obj.iteration + 1},  "
                        f"loss:{loss}, gradient norm: {np.linalg.norm(grad)}, 1/A={1 / A}")
            if save_results:
                outputs = get_outputs(results)
                os.makedirs(checkpoints_path, exist_ok=True)
                save_df(outputs, checkpoints_path)

        lambda_fn_args = get_lambda_fn_args(config=config, obj=obj, lambda_prime_prev=lambda_prime,
                                            lambda_prev=lambda_oracle, prev_A=A)

        grad_norm = norm(obj.grad(x, count_computation=False))

        results.append(dict(
            loss=loss,
            A=A,
            lambda_prime=lambda_prime,
            lambda_oracle_output=lambda_oracle,
            grad_norm=grad_norm,
            norm_x_x_0=norm(x - initial_x),
            norm_x_opt=config.norm_x_opt,
            number_of_hessian_computations=obj.hessian_cnt,
            number_of_linear_system_solves=obj.linear_solves,
            number_of_hvp_computations=obj.hvp_cnt,
            number_of_grad_computations=obj.grad_cnt,
            number_of_fn_computations=obj.fn_cnt,
            timestamp=datetime.datetime.now(),
            t=obj.iteration,
            r=np.linalg.norm(x - y),
            iteration_type="ms_outer_iteration",
            norm_v_x=np.linalg.norm(x - v),
            x_or_z_update=idx,
            test_error=obj.error(x),
            train_error=obj.error(x, test_set=False),
            smoothness=obj.sec_ord_smoothness,
            method_name=opt_ms_algorithm_method_name(config)
        ))

        if np.nan_to_num(loss) > config.stop_loss:
            break

        if grad_norm <= config.min_grad_norm:
            break

    return x, get_outputs(results)


def get_outputs(results):
    dfs = (
        pd.concat([
            pd.DataFrame(parse_results_dict(d), index=[i])
                .rename(columns=dict(t="iteration"))
            for i, d in enumerate(results)
        ])
            .assign(A=lambda df: df.A.fillna(method='ffill'))
    )

    return dfs


def init_results_list(obj: Objective, x: np.array, first_lambda_guess: float, method_name: str):
    return [dict(
        t=0,
        timestamp=datetime.datetime.now(),
        iteration_type="ms_outer_iteration",
        lambda_oracle_output=first_lambda_guess,
        lambda_bisection_output=np.nan,
        lambda_prime=first_lambda_guess,
        lambda_query_oracle=np.nan,
        lambda_query_bisection=np.nan,
        number_of_hessian_computations=obj.hessian_cnt,
        number_of_linear_system_solves=obj.linear_solves,
        number_of_hvp_computations=obj.hvp_cnt,
        number_of_grad_computations=obj.grad_cnt,
        number_of_fn_computations=obj.fn_cnt,
        loss=obj.loss(x),
        grad_norm=norm(obj.grad(x)),
        norm_x_x_0=0,
        A=0,
        method_name=method_name
    )]


def previous_lambda(alpha, lambda_prev):
    lambda_val = alpha * lambda_prev

    return lambda_val


def lower_A_bound(prev_A, norm_x_opt, t, smoothness):
    A_bound = (t / 3) ** (7 / 2) / (2 * smoothness * norm_x_opt)
    a = A_bound - prev_A

    return A_bound / a ** 2


def halving_doubling(lambda_prev, lambda_prime_prev, mul_up=2, mul_down=2):
    if lambda_prev > lambda_prime_prev:
        lambda_prime = lambda_prime_prev * mul_up

    else:
        lambda_prime = lambda_prime_prev / mul_down

    return lambda_prime


def clamped_prev_lambda(lambda_prev, lambda_prime_prev, alpha, mul_up, mul_down):
    if lambda_prev / mul_down > lambda_prime_prev:
        lambda_prime = lambda_prime_prev * mul_up

    elif lambda_prev * mul_up < lambda_prime_prev:
        lambda_prime = lambda_prime_prev / mul_down

    else:
        lambda_prime = alpha * lambda_prev

    return lambda_prime


def get_oracle_fn(config: dataclass):
    return dict(
        amsn=partial(amsn_oracle, lazy=config.lazy_oracle, mul_up=config.mul_up, mul_down=config.mul_down,
                          sigma=config.sigma, lambda_newton=config.lambda_newton),
        cr_oracle=partial(cr_oracle, mul_up=config.mul_up, mul_down=config.mul_down,
                          accuracy=config.cr_oracle_accuracy, lambda_newton=config.lambda_newton),
        amsn_fo_oracle=partial(amsn_fo_oracle, mul_up=config.mul_up, sigma=config.sigma),
    )[config.oracle_type]


LAMBDA_FUNC = dict(
    previous_lambda=previous_lambda,
    lower_A_bound=lower_A_bound,
    halving_doubling=halving_doubling,
    clamped_prev_lambda=clamped_prev_lambda,
)


def get_lambda_fn_args(
        config: MSConfig,
        obj: Objective,
        prev_A,
        lambda_prime_prev: float = None,
        lambda_prev: float = None,
        norm_x_opt: float = None
):
    smoothness = obj.sec_ord_smoothness
    if not norm_x_opt:
        norm_x_opt = config.norm_x_opt
    if not lambda_prev:
        lambda_prev = config.first_lambda_guess
    if not lambda_prime_prev:
        lambda_prime_prev = config.first_lambda_guess

    return dict(
        previous_lambda=dict(alpha=config.alpha, lambda_prev=lambda_prev),
        lower_A_bound=dict(prev_A=prev_A, norm_x_opt=norm_x_opt, t=obj.iteration + 1, smoothness=smoothness),
        halving_doubling=dict(lambda_prev=lambda_prev, lambda_prime_prev=lambda_prime_prev,
                              mul_up=config.alpha, mul_down=config.alpha),
        clamped_prev_lambda=dict(lambda_prev=lambda_prev, lambda_prime_prev=lambda_prime_prev,
                                 alpha=config.alpha, mul_up=config.alpha, mul_down=config.alpha)
    )[config.lambda_func]


def parse_results_dict(results):
    return {k: v for k, v in results.items() if not (isinstance(v, np.ndarray) | (k == 'bisection_results'))}


def opt_ms_algorithm_method_name(config: MSNoBisectionConfig):
    method_name = "opt_ms_algorithm"
    if config.lambda_func == "lower_A_bound":
        method_name = "song_exact_first_bisection"
        if config.exact_first_bisection == False:
            method_name = "song"
    elif config.oracle_type == "cr_oracle":
        method_name = "alg1_cr"
    elif config.oracle_type == "amsn":
        method_name = "alg1_adaptive"
        if config.drop_momentum == True:
            method_name = "iterating_adaptive"
    elif config.oracle_type == "amsn_fo_oracle":
        method_name = "alg1_fo_adaptive"
        if config.drop_momentum == True:
            method_name = "iterating_fo_adaptive"
    return method_name


def ms_algorithm_method_name(config: MSBisectionConfig):
    method_name = "ms_algorithm"
    if config.oracle_type == "cr_oracle":
        method_name = "alg0_cr"
    elif config.oracle_type == "amsn":
        method_name = "alg0_adaptive"
    elif config.oracle_type == "amsn_fo_oracle":
        method_name = "alg0_fo_adaptive"
    return method_name
