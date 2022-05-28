from typing import Union
import numpy as np
import scipy.sparse.linalg


class Objective:
    def __init__(self,
                 features: np.array,
                 labels: np.array,
                 test_features: np.array,
                 test_labels: np.array,
                 objective_type: str,
                 regularization: float = 0,
                 smoothness_mul: float = 1
                 ):
        self.features = features
        self.labels = labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.objective_type = objective_type
        self.regularization = regularization

        self.n = features.shape[0]
        self.test_n = test_features.shape[0]
        self.dim = features.shape[1]

        self.smoothness = self._get_smoothness()
        self.smoothness_mul = smoothness_mul
        self.sec_ord_smoothness = self._get_second_order_smoothness()

        self.fn_cnt = 0
        self.grad_cnt = 0
        self.hvp_cnt = 0
        self.hessian_cnt = 0
        self.linear_solves = 0
        self.iteration = 0
        self.hvp_cnt = 0
        self.grad_cnt = 0
        self.fn_cnt = 0

    def _get_smoothness(self):
        raise NotImplementedError

    def grad(self, x: np.array, count_computation=True):
        raise NotImplementedError

    def hessian(self, x: np.array, count_computation=True):
        raise NotImplementedError

    def hvp_factory(self, x: np.array, count_computation=True):
        raise NotImplementedError

    def loss(self, x, count_computation=True):
        raise NotImplementedError

    def _get_second_order_smoothness(self):
        raise NotImplementedError

    def norm(self, M, **kwargs):
        if scipy.sparse.issparse(M):
            return scipy.sparse.linalg.norm(M, **kwargs)
        else:
            return scipy.linalg.norm(M, **kwargs)

    def solve_linear_system(self, lambda_val, A, b):
        self.linear_solves += 1
        if scipy.sparse.issparse(A) and A.getnnz() / (np.prod(A.shape)) < 0.2:
            return scipy.sparse.linalg.spsolve(A + lambda_val * scipy.sparse.eye(self.dim), b)
        else:
            return np.linalg.solve(A + lambda_val * np.eye(self.dim), b)

    def test_hvp(self, delta=1e-6):
        v = np.random.random(self.dim)
        x = np.random.random(self.dim)
        h = self.hessian(x)
        hvp = self.hvp_factory(x)(v)
        a = h @ v
        cond1 = np.allclose(hvp, a, rtol=1e-8)
        dist1 = np.linalg.norm(hvp - a)

        hvp_fd = (self.grad(x + delta * v) - self.grad(x - delta * v))/(2 * delta)

        cond2 = np.allclose(hvp, hvp_fd, rtol=1e-6)
        dist2 = np.linalg.norm(hvp - hvp_fd)

        return cond1, dist1, cond2, dist2

    def error(self, x, test_set=True):
        if (self.test_n <= 0 and test_set) or not hasattr(self, 'predict'):
            return np.nan
        y_hat = self.predict(x, test_set=test_set)
        if test_set:
            y = self.test_labels
        else:
            y = self.labels
        return np.mean(y_hat != y)

    def first_order_complexity(self):
        return self.fn_cnt + self.grad_cnt + self.hvp_cnt


def neg_linear_loss(
        a: np.array,
        b: np.array,
        x: np.array
) -> Union[float, np.array]:
    """

    Args:
        a:  features (n,d) where n = sample size, d = model dimension.
        b: labels (n,1).
        x: model weights (d,1).

    Returns: the loss: -b * x.T * a

    """
    return -b * (a @ x)