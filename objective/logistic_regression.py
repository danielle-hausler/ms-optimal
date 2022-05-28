from typing import Union

import cvxpy as cp
import numpy as np
from scipy.sparse import diags as sparse_diag
from scipy.special import expit

from objective.objectives import Objective, neg_linear_loss


class LogisticRegression(Objective):
    """
    A class that implements binary Logistic Regression (with regularization).
    """

    def __init__(self, features: np.array, labels: np.array, test_features: np.array, test_labels: np.array,
                 objective_type: str, regularization: float = 0, smoothness_mul=1):
        super().__init__(features, labels, test_features, test_labels, objective_type, regularization, smoothness_mul)

    def _get_smoothness(self):
        return max(self.norm(self.features, axis=1, ord=2)**2) / 4

    def _get_second_order_smoothness(self):
        try:
            b = (self.features.T @ self.features).toarray()
        except:
            b = self.features.T @ self.features
        return self.smoothness_mul * np.max(np.abs(np.linalg.eigh(b)[0])) * max(self.norm(self.features, axis=1)) / self.n

    def grad(self, x: np.array, count_computation=True):
        if count_computation:
            self.grad_cnt += 1
        loss = neg_linear_loss(self.features, self.labels, x)
        g = ((-self.labels * expit(loss)) @ self.features / self.n) + self.regularization * x
        return g

    def hvp_factory(self, x: np.array, count_computation=True):
        loss = neg_linear_loss(self.features, self.labels, x)
        diag_vec = expit(loss) * expit(-loss)

        def hvp(v: np.array):
            if count_computation:
                self.hvp_cnt += 1
            return self.features.T @ (diag_vec * (self.features @ v)) / self.n + self.regularization * v
        return hvp

    def hessian(self, x: np.array, count_computation=True):
        loss = neg_linear_loss(self.features, self.labels, x)
        h = self.features.T @ sparse_diag(expit(loss) * expit(-loss)) @ self.features / self.n
        if count_computation:
            self.hessian_cnt += 1
        return h + self.regularization * np.eye(self.dim)

    def loss(self, x: Union[np.array, cp.Variable], count_computation=True):
        if count_computation:
            self.fn_cnt += 1
        l = neg_linear_loss(self.features, self.labels, x)
        obj = (1 / self.n) * np.sum(np.logaddexp(0, l)) + (self.regularization / 2) * self.norm(x)**2
        return obj

    def regularized_taylor_expansion(self, x, y, lambda_val):
        A = self.hessian(y, count_computation=False)
        b = self.grad(y)
        loss = self.loss(y)

        return loss + np.dot(b, x-y) + (x-y).T @ A @ (x-y) + lambda_val * self.norm(x-y)**2

    def predict(self, x, test_set=True):
        if test_set:
            l = self.test_features @ x
        else:
            l = self.features @ x
        pos_prob = expit(l)
        return np.select([pos_prob > 0.5], [1], default=-1)