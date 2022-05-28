import pdb
from typing import Union, Tuple, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from loguru import logger
import itertools
from numpy.linalg import norm
from scipy.optimize import minimize

from objective.objectives import Objective


def minimize_obj(obj: Objective, exact_solver: str = "lbfgs", max_iter: int = 300):
    """

    Args:
        obj:

    Returns:

    """

    if exact_solver == "lbfgs":
        res = minimize(obj.loss, x0=np.zeros(obj.dim), method="L-BFGS-B", jac=obj.grad)
        x = res.x

    elif exact_solver == "sklearn":
        if obj.objective_type == "logistic_regression":
            clf = LogisticRegression(penalty='none', fit_intercept=False, max_iter=max_iter).fit(
                obj.features, obj.labels)
            x = clf.coef_.flatten()
        else:
            raise ValueError(f"Unsupported objective_type: {obj.objective_type}")

    else:
        raise ValueError(f"Unsupported solver: {exact_solver}")

    return x


def amsn_fo_oracle(
        obj: Objective,
        y: np.array,
        lambda_query: float,
        mul_up: float = 2,
        sigma: float = 0.5,
        ) -> Tuple:

        results = []

        b = -obj.grad(y)
        for k in itertools.count():
            lambda_val = (mul_up ** k) * lambda_query

            hvp = obj.hvp_factory(y, count_computation=True)  # optimistic accounting assuming we use a Krylov subspace method

            def apply_A(v):
                return hvp(v) + lambda_val * v

            w = np.zeros_like(y)
            w0 = w.copy()
            r = apply_A(w) - b
            p = r.copy()
            q = apply_A(r)
            s = q.copy()
            rs = r @ s
            beta = 1.0

            i = 0
            while norm(r) > 0.5 * lambda_val * sigma * norm(w - w0):
                results.append(first_order_oracle_results_summary(obj, lambda_val,
                                                                  k, i, w, w0, r, beta,
                                                                  None, None))
                alpha = rs / (q @ q)
                w -= alpha * p
                r -= alpha * q
                s = apply_A(r)

                rs_next = r @ s
                beta = rs_next / rs
                rs = rs_next

                p *= beta
                p += r
                q *= beta
                q += s

                i += 1

            x = y + w
            grad_x = obj.grad(x)
            results.append(first_order_oracle_results_summary(obj, lambda_val,
                                                              k, i, w, w0, r, beta,
                                                              obj.loss(x, count_computation=False), grad_x,
                                                              ))

            ms_term = np.linalg.norm(w + (1 / lambda_val) * grad_x)
            movement_term = np.linalg.norm(w)
            if ms_term <= sigma * movement_term:
                return x, lambda_val, grad_x, results


def first_order_oracle_results_summary(obj, lambda_val, k, i, w, w0, r, beta, loss_x=None, grad_x=None,
                                       iteration_type='conjugate_residuals_iteration'):
    if grad_x is not None:
        ms_term = np.linalg.norm(w + (1 / lambda_val) * grad_x)
    else:
        ms_term = None
    return dict(
            loss=loss_x,
            lambda_val=lambda_val,
            k=k,
            i=i,
            grad_x_norm=norm(grad_x) if grad_x is not None else None,
            ms_term=ms_term,
            movement_norm=norm(w),
            movement_from_prev_norm=norm(w - w0),
            residual_norm=norm(r),
            beta=beta,
            iteration_type=iteration_type,
            number_of_hvp_computations=obj.hvp_cnt,
            number_of_grad_computations=obj.grad_cnt,
            number_of_fn_computations=obj.fn_cnt,
            t=obj.iteration,
        )


def amsn_oracle(
        obj: Objective,
        y: np.array,
        lambda_query: float,
        lazy: bool = False,
        mul_up: float = 2,
        mul_down: float = 2,
        sigma: float = 0.5,
        lambda_newton: float = 1e-10
        ) -> Tuple:

    results = []

    lambda_vld = None
    x_vld = None

    A = obj.hessian(y)
    b = -obj.grad(y)

    logger.log("TRACE", "starting to run 'ms-oracle' ")
    k=0

    def check_ms(lambda_query, results):
        x = y + obj.solve_linear_system(lambda_query, A, b)
        grad = obj.grad(x)
        ms_term = np.linalg.norm(x - y + (1 / lambda_query) * grad)
        r_term = np.linalg.norm(x - y)
        allowed_ms_error = np.sqrt(sigma ** 2 * r_term ** 2)
        results = update_solver_results(x, lambda_query, obj, r_term, ms_term / allowed_ms_error, "ms", results)
        return x, grad, ms_term <= allowed_ms_error, results

    x, grad, ms_condition, results = check_ms(lambda_query, results)
    if ms_condition or lambda_query < lambda_newton:
        if lazy or lambda_query < lambda_newton:
            return x, lambda_query, grad, results

        while ms_condition and lambda_query >= lambda_newton:
            lambda_vld = lambda_query
            x_vld = x
            grad_vld = grad
            k_star = k

            lambda_query /= mul_down**(mul_down**k)
            x, grad, ms_condition, results = check_ms(lambda_query, results)

            k += 1

        lambda_invld = lambda_vld / mul_down**(mul_down**k_star)

    else:
        while (not ms_condition) and (lambda_query <= 1/lambda_newton):
            lambda_invld = lambda_query
            k_star = k

            lambda_query *= mul_up ** (mul_up ** k)
            x, grad, ms_condition, results = check_ms(lambda_query, results)

            k += 1

        lambda_vld = lambda_invld * mul_up**(mul_up**k_star)
        x_vld = x
        grad_vld = grad

    while lambda_invld < lambda_vld/mul_down and lambda_newton <= lambda_query <= 1/lambda_newton:
        lambda_query = np.sqrt(lambda_vld * lambda_invld)
        x, grad, ms_condition, results = check_ms(lambda_query, results)
        if ms_condition:
            lambda_vld = lambda_query
            x_vld = x
            grad_vld = grad

        else:
            lambda_invld = lambda_query

    x_out = x_vld

    return x_out, lambda_vld, grad_vld, results


def cr_oracle(
        obj: Objective,
        y: np.array,
        lambda_query: float,
        mul_up: float = 2,
        mul_down: float = 2,
        accuracy: float = 1e-10,
        lambda_newton: float = 1e-10,
):
    logger.log("TRACE", "starting to run 'cr-oracle' ")

    upper_c = obj.sec_ord_smoothness * (1 + accuracy)
    lower_c = obj.sec_ord_smoothness * (1 - accuracy)

    results = []
    A = obj.hessian(y)
    b = -obj.grad(y)

    x, r, results = cr_solver_step(obj, y, lambda_query, A, b, results)

    if lower_c * r <= lambda_query <= upper_c * r:
        return x, lambda_query, obj.grad(x), results

    elif lambda_query > upper_c * r:
        lambda_h = lambda_query
        while lambda_query > upper_c * r:
            lambda_query /= mul_down
            x, r, results = cr_solver_step(obj, y, lambda_query, A, b, results, lambda_h=lambda_h)
            if lower_c * r <= lambda_query <= upper_c * r or lambda_query < lambda_newton:
                return x, lambda_query, obj.grad(x), results
        lambda_l = lambda_query
        lambda_mid = np.sqrt(lambda_l * lambda_h)

    else:
        lambda_l = lambda_query
        while lambda_query < lower_c * r:
            lambda_query *= mul_up
            x, r, results = cr_solver_step(obj, y, lambda_query, A, b, results, lambda_l=lambda_l)
            if lower_c * r <= lambda_query <= upper_c * r:
                return x, lambda_query, obj.grad(x), results
        lambda_h = lambda_query
        lambda_mid = np.sqrt(lambda_l * lambda_h)

    done = False
    while not done:
        x, r, results = cr_solver_step(obj, y, lambda_mid, A, b, results, lambda_h=lambda_h, lambda_l=lambda_l)
        if lambda_mid > upper_c * r:
            lambda_h = lambda_mid
        elif lambda_mid < lower_c * r:
            lambda_l = lambda_mid
        else:
            done = True

        lambda_mid = np.sqrt(lambda_l * lambda_h)

    return x, lambda_mid, obj.grad(x), results


def cr_solver_step(obj, y, lambda_query, A, b, results, lambda_h=np.nan, lambda_l=np.nan):
    x = y + obj.solve_linear_system(lambda_query, A, b)
    r = np.linalg.norm(x - y)
    results = update_solver_results(x, lambda_query, obj, r, obj.sec_ord_smoothness * r, "r", results,
                                    lambda_h, lambda_l)
    return x, r, results


def update_solver_results(x: np.array, lambda_val: float, obj: Objective, r: float, oracle_condition: float,
                          oracle_type: str, results: List, lambda_h=np.nan, lambda_l=np.nan):
    results.append(
        dict(
            loss=obj.loss(x),
            lambda_query_oracle=lambda_val,
            grad_norm=np.linalg.norm(obj.grad(x)),
            number_of_hessian_computations=obj.hessian_cnt,
            number_of_linear_system_solves=obj.linear_solves,
            oracle_iteration=len(results),
            t=obj.iteration,
            iteration_type=f"{oracle_type}_oracle_iteration",
            oracle_condition=oracle_condition,
            r=r,
            lambda_h=lambda_h,
            lambda_l=lambda_l
        )
    )
    return results
