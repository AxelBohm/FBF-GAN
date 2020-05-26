import numpy as np


def trunc_decs(value, decs=0):
    return np.trunc(value*10**decs)/(10**decs)

def get_reg_types():
    return ['1norm', '2norm']

def make_prox(rp, reg_type = '1norm'):
    reg_types = get_reg_types()
    if reg_type not in reg_types:
        raise ValueError(f"Invalid regularisation type. Expected one of: {reg_types}")
    def prox(x):
        if reg_type == '1norm':
            return np.sign(x)*np.maximum(np.abs(x)-rp, 0)
        elif reg_type == '2norm':
            return np.multiply(np.maximum(1-rp/np.linalg.norm(x), 0), x)
    return prox

def make_grad(A, a, b, batch_size=1):
    if not isinstance(batch_size, int) or batch_size < 0:
        raise ValueError("Batch size must be a nonnegative integer.")
    def grad(x, y):
        A_stoch = np.zeros_like(A)
        a_stoch = np.zeros_like(a)
        b_stoch = np.zeros_like(b)
        for i in range(batch_size):
            A_stoch += np.random.randn(*A_stoch.shape)
            a_stoch += np.random.randn(*a_stoch.shape)
            b_stoch += np.random.randn(*b_stoch.shape)
        if batch_size > 0:
            A_stoch /= batch_size
            a_stoch /= batch_size
            b_stoch /= batch_size
        A_stoch += A
        a_stoch += a
        b_stoch += b
        return A_stoch @ y + a_stoch, A_stoch.transpose() @ x + b_stoch
    return grad

def make_gap_upper_bound(A, a, b, rp_x, reg_x, rp_y, reg_y):
    reg_types = get_reg_types()
    if not (reg_x in reg_types and reg_y in reg_types):
        raise ValueError(f"Invalid regularisation type. Expected one of: {reg_types}")
    def gap_upper_bound(x, y):
        if reg_x == '1norm':
            test_x = np.linalg.norm(A @ y + a, ord = np.inf)
            regul_x = np.linalg.norm(x, ord = 1)
        elif reg_x == '2norm':
            test_x = np.linalg.norm(A @ y + a, ord = 2)
            regul_x = np.linalg.norm(x, ord = 2)
        if reg_y == '1norm':
            test_y = np.linalg.norm(A.transpose() @ x + b, ord = np.inf)
            regul_y = np.linalg.norm(y, ord = 1)
        elif reg_y == '2norm':
            test_y = np.linalg.norm(A.transpose() @ x + b, ord = 2)
            regul_y = np.linalg.norm(y, ord = 2)
        if not (test_x <= rp_x and test_y <= rp_y):
            return np.inf
        else:
            return np.inner(x,a) - np.inner(y,b) + rp_x*regul_x + rp_y*regul_y
    return gap_upper_bound