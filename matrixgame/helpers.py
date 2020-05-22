import numpy as np

def trunc_decs(value, decs=0):
	return np.trunc(value*10**decs)/(10**decs)

def make_prox(rp, reg_type = '1norm'):
	reg_types = ['1norm', '2norm']
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
