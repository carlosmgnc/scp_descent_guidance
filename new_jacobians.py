import numpy as np
import sympy as sy
from sympy import *

mw = 2.00
md = 1.0
Tmax = 5
Tmin = 0.3
rt = np.array([[-0.01], [0], [0]])
Jbvec = np.array([[0.01], [0.01], [0.01]])
Jb = np.diag(Jbvec.flatten())
Jbinv = np.diag((1 / Jbvec).flatten())
alpha = 0.1

nk = 50
K = np.arange(0, nk)
dt = 1 / (nk - 1)
tau = np.linspace(0, 1, nk)

# initial conditions
ri = np.array([[4], [4], [0]])
vi = np.array([[-0.5], [-0.7], [0]])
vf = np.array([[0], [0], [0]])
g = np.array([[-1], [0], [0]])

tfguess = 10

# define initial guess trajectory (wow, i didn't know the guess trajectory could be so simple)
# alpha1_list = (nk - (K + 1)) / nk
# alpha2_list = (K + 1) / nk

alpha1_list = (1 - tau) / (1)
alpha2_list = 1 - alpha1_list

mk = alpha1_list * mw + alpha2_list * md
rk = alpha1_list * ri
vk = alpha1_list * vi + alpha2_list * vf
qk = np.vstack([np.ones((1, nk)), np.zeros((3, nk))])
wk = np.zeros((3, nk))

xk = np.vstack([mk, rk, vk, qk, wk])


def DCM(q):
    return sy.Matrix(
        [
            [
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] + q[0] * q[3]),
                2 * (q[1] * q[3] - q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] - q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] + q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] + q[0] * q[2]),
                2 * (q[2] * q[3] - q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ],
        ]
    )


def omega(w):
    return sy.Matrix(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ]
    )


def cr(v):
    return sy.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


f_symb = sy.zeros(14, 1)

x = sy.Matrix(sy.symbols("m r0 r1 r2 v0 v1 v2 q0 q1 q2 q3 w0 w1 w2", real=True))
u = sy.Matrix(sy.symbols("u0 u1 u2", real=True))

g1 = np.array([[-1], [0], [0]])
rt = np.array([[-0.01], [0], [0]])
Jbvec = np.array([[0.01], [0.01], [0.01]])
Jb = np.diag(Jbvec.flatten())
Jbinv = np.diag((1 / Jbvec).flatten())

g = sy.Matrix(g1)
rt = sy.Matrix(rt)
Jb = sy.Matrix(Jb)
Jbinv = sy.Matrix(Jbinv)
alpha = 0.1

# f_symb[0, 0] = -alpha * u.norm()
f_symb[1:4, 0] = x[4:7, 0]
# f_symb[4:7, 0] = (1/x[0, 0]) * DCM(x[7:11, 0]).T * u + g
# f_symb[7:11, 0] = (1/2) * omega(x[11:14, 0]) * x[7:11, 0]
# f_symb[11:14, 0] = Jbinv * (cr(rt) * u - cr(x[11:14, 0]) * Jb * x[11:14, 0])

f_symb[:1, 0] = sy.zeros(1, 1)
f_symb[4:14, 0] = sy.zeros(10, 1)

print(f_symb)

f_symb = f_symb
A_symb = f_symb.jacobian(x)
B_symb = f_symb.jacobian(u)
z_symb = -A_symb * x - B_symb * u

A_f = sy.lambdify([x, u], A_symb, "numpy")
B_f = sy.lambdify([x, u], B_symb, "numpy")
E_f = sy.lambdify([x, u], f_symb, "numpy")
z_f = sy.lambdify([x, u], z_symb, "numpy")


ones14 = np.ones((14, 1))
u1 = np.ones((3, 1))



nsq = 14 * 14
P_temp = np.empty((9 * 14 + nsq, 1))

P_temp[:14, [0]] = xk[:, [0]]
P_temp[14 : 14 + nsq, [0]] = np.eye(14).reshape((nsq, 1))
P_temp[14 + nsq : 4 * 14 + nsq, [0]] = np.zeros((14, 3)).reshape((14 * 3, 1))
P_temp[4 * 14 + nsq : 7 * 14 + nsq, [0]] = np.zeros((14, 3)).reshape((14 * 3, 1))
P_temp[7 * 14 + nsq : 8 * 14 + nsq, [0]] = np.zeros((14, 1))
P_temp[8 * 14 + nsq : 9 * 14 + nsq, [0]] = np.zeros((14, 1))

uk = -mk * g1

print(E_f(P_temp[:14].flatten(), u1.flatten()))
