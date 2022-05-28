import numpy as np
import scipy.sparse
from scipy.sparse import diags as sparse_diag

from objective.objectives import Objective


class GenChain(Objective):
    def __init__(self, dim, p=3,
                 features=np.empty((0,0)), labels=np.empty(0), test_features=np.empty(0), test_labels=np.empty(0),
                 objective_type="gen_chain", regularization=0):
        super().__init__(features, labels, test_features, test_labels, objective_type, regularization)
        self.dim = dim
        self.p = p

        self.sec_ord_smoothness = None

        self.A = scipy.sparse.csr_matrix(np.eye(dim) - np.diag(np.ones(dim-1), -1))
        self.b = np.zeros(dim)
        self.b[0] = - 1

    def psi(self, x):
        return np.abs(x) ** self.p

    def dpsi(self, x):
        return self.p * (np.abs(x) ** (self.p - 1)) * np.sign(x)

    def ddpsi(self, x):
        return self.p * (self.p - 1) * (np.abs(x) ** (self.p - 2))

    def loss(self, x, count_computation=True):
        if count_computation:
            self.fn_cnt += 1
        return np.sum(self.psi(self.A @ x - self.b))

    def grad(self, x, count_computation=True):
        if count_computation:
            self.grad_cnt += 1
        return self.A.T @ self.dpsi(self.A @ x - self.b)

    def hessian(self, x, count_computation=True):
        if count_computation:
            self.hessian_cnt += 1
        return self.A.T @ sparse_diag(self.ddpsi(self.A @ x - self.b)) @ self.A

    def _get_smoothness(self):
        pass

    def hvp_factory(self, x: np.array, count_computation=True):
        pass

    def _get_second_order_smoothness(self):
        pass

    def test_hvp(self, delta=1e-6):
        v = np.random.random(self.dim)
        x = np.random.random(self.dim)
        h = self.hessian(x)
        a = h @ v
        gv = self.grad(x) @ v

        grad_fd = (self.loss(x + delta * v) - self.loss(x - delta * v))/(2 * delta)

        cond1 = np.allclose(gv, grad_fd, rtol=1e-6)
        dist1 = np.linalg.norm(gv - grad_fd)

        hvp_fd = (self.grad(x + delta * v) - self.grad(x - delta * v))/(2 * delta)

        cond2 = np.allclose(a, hvp_fd, rtol=1e-6)
        dist2 = np.linalg.norm(a - hvp_fd)

        return cond1, dist1, cond2, dist2