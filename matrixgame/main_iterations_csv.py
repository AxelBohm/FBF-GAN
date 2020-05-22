import numpy as np
from helpers import *
from RIFBF import RIFBF_bilin_const

#rand_seed = 42
rand_seed = 9001
np.random.seed(seed=rand_seed)

# need square sub-matrix to get full rank M
n = 500
m = n
N = n+m
	
A = np.random.rand(m, n)
# print(np.linalg.matrix_rank(A))

M = np.block([
	[np.zeros((m,m)), A],
	[-A.transpose(),  np.zeros((n,n))]
])
# print(np.linalg.matrix_rank(M))

L = np.linalg.norm(M)
# print(L)

a = np.random.rand(m,1)
b = np.random.rand(n,1)
q = np.append(a,-b, axis = 0)


mu_vec = [0.9, 0.5, 0.1]
n_max = 1e4
tol = 1e-5
x_0 = np.random.rand(N,1)

# TODO: maybe also save array of residuals to judge performance for iter > n_max!

for mu in mu_vec:
	print("\nmu = {}\n".format(mu))

	rm = rho_max(0, mu)
	rho = np.arange(start = 0, stop = rm, step = 0.1)
	# print(rm)
	rho[0] = 0.01
	rho = np.append(rho, trunc_decs(rm, decs=2) - 0.01)
	rho = np.unique(rho)
	# print(rho)

	alpha = np.arange(start = 0, stop = alpha_max(rho[0], mu), step = 0.04)
	# alpha = np.arange(start = 0, stop = 1., step = 0.02)
	# alpha = np.append(alpha, 1.)
	# print(alpha)

	arr = np.empty((len(alpha), len(rho)))
	arr.fill(-1)
	arr.astype(int)
	for i, a in enumerate(alpha):
		for j, r in enumerate(rho):
			print("alpha = {};\trho = {}".format(a,r))
			if a > alpha_max(r, mu):
				break
			else:
				_, iter, _, _ = RIFBF_bilin_const(M = M, q = q, dim = n, proj_abb = "bl", x0 = x_0, alpha = a,
											rho = r, lmbd = mu/L, max_iter = n_max, eps = tol)
				arr[i,j] = iter
	# print(arr)

	arr = np.append(np.reshape(alpha, (len(alpha),1)), arr, axis = 1)
	arr = np.append(np.reshape(np.append(0., rho), (1,len(rho)+1)), arr, axis = 0)
	# print(arr)

	np.savetxt("./tables/Number_iterations_{}_mu-{}.csv".format(rand_seed, mu), arr, delimiter = ",", fmt = "%.2f")
