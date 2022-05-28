from loguru import logger
import numpy as np
from functools import partial
from objective.objectives import Objective


def ms_bisection(
        obj: Objective,
        x: np.array,
        v: np.array,
        A: float,
        lambda_query: float,
        oracle: callable,
        mul_up: float = 2,
        mul_down: float = 2,
        lower_rho: float = 1,
        upper_rho: float = 2.01,
        max_iterations: int = 1000
):
    logger.log("TRACE", f"starting bisection")

    y = partial(calc_y, x, v, A)
    is_valid_bisection = partial(basic_bisection_condition, lower_c=lower_rho, upper_c=upper_rho)

    results = []
    i = 0

    lambda_mid = lambda_query
    x_next, lambda_oracle, grad, oracle_results = oracle(obj, y(lambda_mid), lambda_mid)
    results = update_results(results, lambda_mid, x_next, lambda_oracle, oracle_results, obj, i)

    if is_valid_bisection(lambda_mid, lambda_oracle):
        return x_next, lambda_oracle, lambda_mid, grad, results

    elif lambda_query > upper_rho * lambda_oracle:
        lambda_h = lambda_query
        lambda_l_oracle = lambda_oracle
        while lower_rho * lambda_l_oracle <= lambda_query and i <= max_iterations:
            i += 1
            lambda_query /= mul_down
            x_next, lambda_l_oracle, grad, oracle_results = oracle(obj, y(lambda_query), lambda_query)
            results = update_results(results, lambda_query, x_next, lambda_l_oracle, oracle_results, obj, i,
                                     lambda_h=lambda_h)

            if is_valid_bisection(lambda_query, lambda_l_oracle):
                return x_next, lambda_l_oracle, lambda_query, grad, results

        lambda_l = lambda_query


    else:
        lambda_l = lambda_query
        lambda_h_oracle = lambda_oracle
        while lambda_query <= lower_rho * lambda_h_oracle and i <= max_iterations:
            i += 1
            lambda_query *= mul_up
            x_next, lambda_h_oracle, grad, oracle_results = oracle(obj, y(lambda_query), lambda_query)
            results = update_results(results, lambda_query, x_next, lambda_h_oracle, oracle_results, obj, i,
                                     lambda_l=lambda_l)
            if is_valid_bisection(lambda_query, lambda_h_oracle):
                return x_next, lambda_h_oracle, lambda_query, grad, results
        lambda_h = lambda_query

    while (not is_valid_bisection(lambda_mid, lambda_oracle)) and i <= max_iterations:
        i += 1
        logger.log("TRACE",
                   f" bisection iteration number {i}, lambda oracle = {lambda_oracle}, lambda_mid = {lambda_mid}")
        if lambda_mid > upper_rho * lambda_oracle:
            lambda_h = lambda_mid
        else:
            lambda_l = lambda_mid

        lambda_mid = np.sqrt(lambda_l * lambda_h)

        x_next, lambda_oracle, grad, oracle_results = oracle(obj, y(lambda_mid), lambda_mid)
        results = update_results(results, lambda_mid, x_next, lambda_oracle, oracle_results, obj, i, lambda_h, lambda_l)

    return x_next, lambda_oracle, lambda_mid, grad, results


def basic_bisection_condition(lambda_query, lambda_oracle, lower_c, upper_c):
    return lower_c * lambda_oracle <= lambda_query <= upper_c * lambda_oracle


def update_results(
        results,
        lambda_query,
        x_next,
        lambda_oracle,
        oracle_results,
        obj: Objective,
        i,
        lambda_h=np.nan,
        lambda_l=np.nan
):
    results.extend([{**d,
                     "bisection_iteration": i,
                     'lambda_query_bisection': lambda_query,
                     'lambda_h': lambda_h,
                     'lambda_l': lambda_l,
                     }
                    for d in oracle_results])

    results.append({**results[-1],
                    'lambda_oracle_output': lambda_oracle,
                    'lambda_h': lambda_h,
                    'lambda_l': lambda_l,
                    'oracle_iteration': np.nan,
                    'lambda_query_oracle': np.nan,
                    'oracle_condition': np.nan,
                    'r': np.nan,
                    'loss': obj.loss(x_next),
                    'iteration_type': 'bisection_iteration'})
    return results


def calc_y(x, v, A, lambda_val):
    a = (1 / (2 * lambda_val)) * (1 + np.sqrt(1 + (4 * lambda_val * A)))
    A_next = A + a

    return ((A / A_next) * x) + ((a / A_next) * v)

