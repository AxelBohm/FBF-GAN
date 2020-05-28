import numpy as np

def trunc_decs(value, decs=0):
    return np.trunc(value*10**decs)/(10**decs)

def get_reg_types():
    return ['1norm', '2norm']

def check_reg_type(reg_type):
    if reg_type not in get_reg_types():
        raise ValueError(f"Invalid regularisation type. Expected one of: {get_reg_types()}")

def regul(x, rp, reg_type):
    check_reg_type(reg_type)
    if reg_type == '1norm':
        return rp*np.linalg.norm(x, ord = 1)
    elif reg_type == '2norm':
        return rp*np.linalg.norm(x, ord = 2)

def make_prox(rp, reg_type = '1norm'):
    check_reg_type(reg_type)
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
    check_reg_type(reg_x)
    check_reg_type(reg_y)
    def gap_upper_bound(x, y):
        if reg_x == '1norm':
            test_x = np.linalg.norm(A @ y + a, ord = np.inf)
        elif reg_x == '2norm':
            test_x = np.linalg.norm(A @ y + a, ord = 2)
        if reg_y == '1norm':
            test_y = np.linalg.norm(A.transpose() @ x + b, ord = np.inf)
        elif reg_y == '2norm':
            test_y = np.linalg.norm(A.transpose() @ x + b, ord = 2)
        if not (test_x <= rp_x and test_y <= rp_y):
            return np.inf
        else:
            return np.inner(x,a) - np.inner(y,b) + regul(x, rp_x, reg_x) + regul(y, rp_y, reg_y)
    return gap_upper_bound

def make_gap(A, a, b, rp_x, reg_x, rp_y, reg_y, x_sol, y_sol, radius):
    check_reg_type(reg_x)
    check_reg_type(reg_y)
    def gap(x, y):
        const = np.inner(x,a) - np.inner(y,b) + regul(x, rp_x, reg_x) + regul(y, rp_y, reg_y)
        inf_reg_x = solve_inf_reg(rp_x, reg_x, A.transpose() @ x + b, x_sol, radius)
        inf_reg_y = solve_inf_reg(rp_y, reg_y, np.multiply(-1., A @ y + a), y_sol, radius)
        return const + inf_reg_x + inf_reg_y
    return gap

def solve_inf_reg(rp, reg_type, vec, sol, r):
    check_reg_type(reg_type)
    n_iter = 400
    lr = 2.
    
    def prox_f(x):
        tmp_prox = make_prox(r*lr, '2norm')
        return tmp_prox(x - np.multiply(lr, sol))
    
    def prox_g(x):
        if reg_type == '1norm':
            ord = np.inf
        elif reg_type == '2norm':
            ord = 2
        tmp_vec = x - vec
        tmp_norm = np.linalg.norm(tmp_vec, ord = ord)
        if tmp_norm <= rp:
            return x
        else:
            return vec + np.multiply(rp/tmp_norm, tmp_vec)
    
    x = vec
    for i in range(n_iter):
        y = prox_g(x)
        z = prox_f(np.multiply(2., y) - x)
        x = x + z - y
    
    arg_min = prox_g(x)
    
    return np.inner(arg_min, sol) + r*np.linalg.norm(arg_min)