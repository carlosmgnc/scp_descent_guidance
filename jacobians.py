import numpy as np
import sympy as sy
from sympy import *

# state vector variables
m = Symbol("m")
r = MatrixSymbol("r", 3, 1)
v = MatrixSymbol("v", 3, 1)
q = MatrixSymbol("q", 4, 1)
w = MatrixSymbol("w", 3, 1)

# input
u = MatrixSymbol("u", 3, 1)

rt = MatrixSymbol("rt", 3, 1)

# inertia matrix (asymmetric)
# Jb = MatrixSymbol('Jb', 3, 3)
# Jbinv = MatrixSymbol('Jbinv', 3, 3)

# inertia matrix (symmetric)
Jb_vec = MatrixSymbol("Jb_vec", 3, 1)
Jb = diag(Jb_vec[0, 0], Jb_vec[1, 0], Jb_vec[2, 0])
Jbinv = diag(1 / Jb_vec[0, 0], 1 / Jb_vec[1, 0], 1 / Jb_vec[2, 0])

rt_cr = Matrix([[0, -rt[2], rt[1]], [rt[2], 0, -rt[0]], [-rt[1], rt[0], 0]])

w_cr = Matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

C = Matrix(
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

omega = Matrix(
    [
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0],
    ]
)

v_dot = Matrix((1 / m) * C.T * u)
q_dot = Matrix((1 / 2) * omega * q)
w_dot = Matrix(Jbinv * (rt_cr * u - w_cr * Jb * w))
# for calculting partial jacobian for asymmetric Jb
# w_dot_J = Matrix((rt_cr*u - w_cr*Jb*w))

D_vq = v_dot.jacobian(q)
D_qw = q_dot.jacobian(w)
D_ww = w_dot.jacobian(w)
# D_ww_J = w_dot_J.jacobian(w)


print(D_vq)
print()
print(D_qw)
print()
print(D_ww)
print()
print(D_vq.shape)
print(D_qw.shape)
print(D_ww.shape)
