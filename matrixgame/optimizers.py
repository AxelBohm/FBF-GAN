import numpy as np
import timeit
from helpers import make_grad, make_prox

def FBF(A, a, b, rp_x, rp_y, x0, y0, LR, reg_x='1norm', reg_y='2norm', batch_size=10, max_iter=1e04, eps=1e-5):
	grad = make_grad(A, a, b, batch_size)
	prox_x = make_prox(LR*rp_x, reg_x)
	prox_y = make_prox(LR*rp_y, reg_y)

	# def gap(theta, phi):
	#	infimum = - np.linalg.norm(A @ phi + a) + np.dot(b, np.squeeze(phi))
	#	supremum = np.linalg.norm(A.transpose() @ theta + b) + np.dot(a, np.squeeze(theta))
	#	return np.abs(infimum - supremum)

	k = 0
	Error_fp = []
	Error_gap = []
	res_fp = None
	res_gap = None

	t0 = timeit.default_timer()
	x, y = x0, y0
	while k < max_iter:
		k+=1
		g_x, g_y = grad(x, y)
		u = prox_x(x - LR*g_x)
		v = prox_y(y + LR*g_y)

		res_fp = np.sqrt(np.linalg.norm(x-u)**2 + np.linalg.norm(y-v)**2)
		# print(res_fp)
		Error_fp.append(res_fp)

		g_u, g_v = grad(u, v)
		x = u - LR*g_u + LR*g_x
		y = v + LR*g_v - LR*g_y

		# res_gap = gap(x, y)
		# Error_gap.append(res_gap)

		if not k%1000:
			print(k, "\t: ", res_fp)

		if res_fp <= eps:
			break

	t = timeit.default_timer() - t0

	# report results
	print("\nNumber of iterations = ", k)
	print("Total time = ", t)
	print("Residual_fp = ", res_fp)
	print("Residual_gap = ", res_gap, "\n")

	print("Norm x: ", np.linalg.norm(x))
	print("Norm y: ", np.linalg.norm(y))

	return x, y, k, t, Error_fp, Error_gap

def RIFBF_const_step(func, x0, x1, alpha, rho, lmbd, max_iter, eps):
	# initial iter and parameter
	k = 0
	Error = []
	
	x_old = x0
	x = x1
	
	begin = timeit.default_timer()
	
	while  k < max_iter:
		k+=1
		z = x + alpha*(x - x_old)
		Fz = func(z)
		y = P_nncone(z - lmbd*Fz)
		
		# if residual == 0 then y is a solution
		res = np.linalg.norm(y-z)
		Error.append(res)
		if res <= eps:
			x = y
			break
		
		# compute new iterate
		x_new = (1-rho)*z + rho*(y - lmbd*(func(y) - Fz))
		
		# update parameters
		x_old = x
		x = x_new
		

	end = timeit.default_timer()
	time = end - begin
	
	# report results
	# print("\nTotal time = ", time)
	# print("Number of iterations = ", k)
	# print("Residual = ", res, "\n")

	return x, k, time, Error




