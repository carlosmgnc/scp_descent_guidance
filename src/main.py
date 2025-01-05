import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# class sim:
#   def __init__(self):


class opt_problem:
    def __init__(self):

        # vehicle properties
        self.mw = 2.00
        self.md = 1.0
        self.Tmax = 5
        self.Tmin = 0.3
        self.rt = np.array([[-0.01], [0], [0]])
        self.Jbvec = np.array([[0.01], [0.01], [0.01]])
        self.Jb = np.diag(self.Jbvec.flatten())
        self.Jbinv = np.diag((1 / self.Jbvec).flatten())
        self.alpha = 0.1

        self.nk = 50
        self.K = np.arange(0, self.nk)
        self.dt = 1 / (self.nk - 1)
        self.tau = np.linspace(0, 1, self.nk)

        # initial conditions
        ri = np.array([[4], [4], [0]])
        vi = np.array([[-0.5], [-1], [0]])
        vf = np.array([[0], [0], [0]])
        self.g = np.array([[-1], [0], [0]])

        tfguess = 1

        # define initial guess trajectory (wow, i didn't know the guess trajectory could be so simple)
        alpha1_list = (self.nk - (self.K + 1)) / self.nk
        alpha2_list = (self.K + 1) / self.nk

        self.mk = alpha1_list * self.mw + alpha2_list * self.md
        self.rk = alpha1_list * ri
        self.vk = alpha1_list * vi + alpha2_list * vf
        self.qk = np.vstack([np.ones((1, self.nk)), np.zeros((3, self.nk))])
        self.wk = np.zeros((3, self.nk))

        # print(self.rk)
        self.xk = np.vstack([self.mk, self.rk, self.vk, self.qk, self.wk])
        self.uk = -self.mk * self.g

        self.A = np.empty((self.nk - 1, 14, 14))
        self.B = np.empty((self.nk - 1, 14, 3))
        self.C = np.empty((self.nk - 1, 14, 3))
        self.E = np.empty((14, self.nk - 1))
        self.z = np.empty((14, self.nk - 1))
        self.sigmak = tfguess

        self.stm_inv = np.zeros((14, 14))

    # compute the LTV model for the guess trajectory by taylor series

    def linearize(self, t, return_value="a"):
        k = int(np.floor(t / self.dt))
        tk = self.tau[k]

        if t != 1:
            tk1 = self.tau[k + 1]
            alphak = (tk1 - t) / (tk1 - tk)
            betak = 1 - alphak
            x = alphak * self.xk[:, k] + betak * self.xk[:, k + 1]
            u = alphak * self.uk[:, k] + betak * self.uk[:, k + 1]
        else:
            alphak = 1
            betak = 0
            x = alphak * self.xk[:, k]
            u = alphak * self.uk[:, k]

        if return_value == "alphak":
            return alphak

        if return_value == "betak":
            return betak

        # interpolated trajectory point
        m = x[0]
        r = x[1:4]
        v = x[4:7]
        q = x[7:11]
        w = x[11:14]

        C = np.array(
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

        omega = np.array(
            [
                [0, -w[0], -w[1], -w[2]],
                [w[0], 0, w[2], -w[1]],
                [w[1], -w[2], 0, w[0]],
                [w[2], w[1], -w[0], 0],
            ]
        )

        rt_cr = np.array(
            [
                [0, -self.rt[2, 0], self.rt[1, 0]],
                [self.rt[2, 0], 0, -self.rt[0, 0]],
                [-self.rt[1, 0], self.rt[0, 0], 0],
            ]
        )

        w_cr = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

        # jacobian w.r.t state
        # respective jacobians obtained with sympy
        if return_value == "A" or return_value == "z":
            D_vq = np.array(
                [
                    [
                        2 * q[2] * u[2] / m - 2 * q[3] * u[1] / m,
                        2 * q[2] * u[1] / m + 2 * q[3] * u[2] / m,
                        2 * q[0] * u[2] / m + 2 * q[1] * u[1] / m - 4 * q[2] * u[0] / m,
                        -2 * q[0] * u[1] / m
                        + 2 * q[1] * u[2] / m
                        - 4 * q[3] * u[0] / m,
                    ],
                    [
                        -2 * q[1] * u[2] / m + 2 * q[3] * u[0] / m,
                        -2 * q[0] * u[2] / m
                        - 4 * q[1] * u[1] / m
                        + 2 * q[2] * u[0] / m,
                        2 * q[1] * u[0] / m + 2 * q[3] * u[2] / m,
                        2 * q[0] * u[0] / m + 2 * q[2] * u[2] / m - 4 * q[3] * u[1] / m,
                    ],
                    [
                        2 * q[1] * u[1] / m - 2 * q[2] * u[0] / m,
                        2 * q[0] * u[1] / m - 4 * q[1] * u[2] / m + 2 * q[3] * u[0] / m,
                        -2 * q[0] * u[0] / m
                        - 4 * q[2] * u[2] / m
                        + 2 * q[3] * u[1] / m,
                        2 * q[1] * u[0] / m + 2 * q[2] * u[1] / m,
                    ],
                ]
            )

            D_qw = np.array(
                [
                    [-0.5 * q[1], -0.5 * q[2], -0.5 * q[3]],
                    [0.5 * q[0], -0.5 * q[3], 0.5 * q[2]],
                    [0.5 * q[3], 0.5 * q[0], -0.5 * q[1]],
                    [-0.5 * q[2], 0.5 * q[1], 0.5 * q[0]],
                ]
            )

            D_ww = np.array(
                [
                    [
                        0,
                        (self.Jbvec[1, 0] * w[2] - self.Jbvec[2, 0] * w[2])
                        / self.Jbvec[0, 0],
                        (self.Jbvec[1, 0] * w[1] - self.Jbvec[2, 0] * w[1])
                        / self.Jbvec[0, 0],
                    ],
                    [
                        (-self.Jbvec[0, 0] * w[2] + self.Jbvec[2, 0] * w[2])
                        / self.Jbvec[1, 0],
                        0,
                        (-self.Jbvec[0, 0] * w[0] + self.Jbvec[2, 0] * w[0])
                        / self.Jbvec[1, 0],
                    ],
                    [
                        (self.Jbvec[0, 0] * w[1] - self.Jbvec[1, 0] * w[1])
                        / self.Jbvec[2, 0],
                        (self.Jbvec[0, 0] * w[0] - self.Jbvec[1, 0] * w[0])
                        / self.Jbvec[2, 0],
                        0,
                    ],
                ]
            )

            u = u.reshape(3, 1)
            w = w.reshape(3, 1)

            D_vm = -(C.T @ u) / (m**2)
            D_qq = 0.5 * omega

            D_x1 = np.zeros((1, 14))
            D_x2 = np.hstack([np.zeros((3, 4)), np.eye(3), np.zeros((3, 7))])
            D_x3 = np.hstack([D_vm, np.zeros((3, 6)), D_vq, np.zeros((3, 3))])
            D_x4 = np.hstack([np.zeros((4, 7)), D_qq, D_qw])
            D_x5 = np.hstack([np.zeros((3, 11)), D_ww])
            D_x = np.vstack([D_x1, D_x2, D_x3, D_x4, D_x5])

            Ac = self.sigmak * D_x

            if return_value == "A":
                return Ac

        u = u.reshape(3, 1)
        v = v.reshape(3, 1)
        q = q.reshape(4, 1)
        w = w.reshape(3, 1)
        x = x.reshape((14, 1))

        if return_value == "B" or return_value == "z":
            # jacobian w.r.t input
            D_u = np.vstack(
                [
                    -u.T * (self.alpha) / np.linalg.norm(u),
                    np.zeros((3, 3)),
                    C.T / m,
                    np.zeros((4, 3)),
                    self.Jbinv @ rt_cr,
                ]
            )

            Bc = self.sigmak * D_u
            if return_value == "B":
                return Bc

        if return_value == "E":
            # nonlinear propagation term
            f1 = np.array([[-self.alpha * np.linalg.norm(u)]])
            f2 = v
            f3 = (1 / m) * C.T @ u + self.g
            f4 = 0.5 * omega @ q
            f5 = self.Jbinv @ (rt_cr @ u - w_cr @ self.Jb @ w)

            f2 = f2.reshape((3, 1))
            f4 = f4.reshape((4, 1))
            Ec = np.vstack([f1, f2, f3, f4, f5])
            return Ec

        if return_value == "z":
            zc = -Ac @ x - Bc @ u
            return zc

    # rk4 for discretization

    def stm_func(self, t, x):
        return self.linearize(t, "A") @ x

    def B_func(self, t, x):
        return self.stm_inv @ self.linearize(t, "B") * self.linearize(t, "alphak")

    def C_func(self, t, x):
        return self.stm_inv @ self.linearize(t, "B") * self.linearize(t, "betak")

    def E_func(self, t, x):
        return self.stm_inv @ self.linearize(t, "E")

    def z_func(self, t, x):
        return self.stm_inv @ self.linearize(t, "z")

    def rk41(self, func, tk, xk, dt):
        k1 = func(tk, xk)
        k2 = func(tk + dt / 2, xk + (dt / 2) * k1)
        k3 = func(tk + dt / 2, xk + (dt / 2) * k2)
        k4 = func(tk + dt, xk + dt * k3)
        output = xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return output

    def discretize(self):
        nsub = 15
        dt_sub = self.dt / (nsub + 1)

        A_temp = np.eye(14)
        B_temp = np.zeros((14, 3))
        C_temp = np.zeros((14, 3))
        E_temp = np.zeros((14, 1))
        z_temp = np.zeros((14, 1))

        for i in range(0, self.nk - 1):
            print("i: " + str(i))
            for j in range(0, nsub + 1):
                sub_time = i * self.dt + j * dt_sub
                print("alpha: " + str(self.linearize(sub_time, "alphak")))
                A_temp = self.rk41(self.stm_func, sub_time, A_temp, dt_sub/2)
                self.stm_inv = np.linalg.inv(A_temp)
                A_temp = self.rk41(self.stm_func, sub_time, A_temp, dt_sub/2)
                B_temp = A_temp @ self.rk41(self.B_func, sub_time, B_temp, dt_sub)
                C_temp = A_temp @ self.rk41(self.C_func, sub_time, C_temp, dt_sub)
                E_temp = A_temp @ self.rk41(self.E_func, sub_time, E_temp, dt_sub)
                z_temp = A_temp @ self.rk41(self.z_func, sub_time, z_temp, dt_sub)
                # print(sub_time)
            self.A[i, :, :] = A_temp
            self.B[i, :, :] = B_temp
            self.C[i, :, :] = C_temp
            self.E[:, [i]] = E_temp
            self.z[:, [i]] = z_temp

            A_temp = np.eye(14)
            B_temp = np.zeros((14, 3))
            C_temp = np.zeros((14, 3))
            E_temp = np.zeros((14, 1))
            z_temp = np.zeros((14, 1))

        # array = self.A[48,:, :]
        # formatted_array = np.array([[f"{x:.4g}" for x in row] for row in array])
        # print("\n".join(["\t".join(row) for row in formatted_array]))

    def solve_cvx_problem(self):
        x = cvx.Variable((14, self.nk))
        u = cvx.Variable((3, self.nk))
        sigma = cvx.Variable()

        # soft trust region variables
        delta = cvx.Variable((self.nk, 1), nonneg=True)
        deltasigma = cvx.Variable(nonneg=True)

        # artificial control variable
        nu = cvx.Variable((14, self.nk - 1), nonneg=True)

        w_delta = 0.001
        w_deltasigma = 0.01
        w_nu = 100000  ##

        theta_max = np.deg2rad(90)
        deltamax = np.deg2rad(20)
        w_max = np.deg2rad(60)

        cost = (
            sigma
            + w_delta
            + w_nu * cvx.norm(nu)
            + w_delta * cvx.norm(delta)
            + w_deltasigma * cvx.norm(deltasigma, 1)
        )
        constraints = []
        # boundary constraints
        constraints += [x[:, 0] == self.xk[:, 0]]
        constraints += [x[:, -1] == self.xk[:, -1]]
        constraints += [x[1, :] >= 0]

        for k in range(0, self.nk - 1):
            constraints += [
                x[:, [k + 1]]
                == self.A[k, :, :] @ x[:, [k]]
                + self.B[k, :, :] @ u[:, [k]]
                + self.C[k, :, :] @ u[:, [k + 1]]
                + self.E[:, [k]] * sigma
                + self.z[:, [k]]
                + nu[:, [k]]
            ]

        # state inequality constraints
        constraints += [self.md <= x[0, :]]
        for k in range(0, self.nk):
            H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
            constraints += [np.cos(theta_max) <= 1 - 2 * cvx.norm(H @ x[7:11, [k]])]
            constraints += [cvx.norm(x[11:14, [k]]) <= w_max]

        # constraints += []

        # control constraints
        for k in range(0, self.nk):
            constraints += [cvx.norm(u[:, [k]]) <= self.Tmax]
            constraints += [
                self.Tmin
                <= (np.transpose(self.uk[:, [k]]) / np.linalg.norm(self.uk[:, [k]]))
                @ u[:, [k]]
            ]
            constraints += [np.cos(deltamax) * cvx.norm(u[:, [k]]) <= u[0, [k]]]

            constraints += [u[0, k] >= 0]

        # trust region
        for k in range(0, self.nk):
            dx = x[:, [k]] - self.xk[:, [k]]
            du = u[:, [k]] - self.uk[:, [k]]
            constraints += [delta[k] >= 0]
            constraints += [cvx.norm(cvx.vstack([dx, du])) <= cvx.sqrt(delta[k])]
            # print (cvx.vstack([dx, du]).shape)
            # constraints += [dx.T @ dx + du.T @ du <= delta[k]]

        dsigma = sigma - self.sigmak
        constraints += [cvx.abs(dsigma) <= cvx.sqrt(deltasigma)]

        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective, constraints)
        print("going in to solve")
        prob.solve("CLARABEL")
        print("solver status: " + prob.status)
        print(prob.solver_stats.solve_time)
        print(prob.solver_stats.solver_name)

        self.xk = x.value
        self.uk = u.value
        self.sigmak = sigma.value

        return x.value, u.value, nu.value


opt = opt_problem()

for i in range(0, 3):
    opt.discretize()
    x, u, nu = opt.solve_cvx_problem()


##### plotting #####


plt.figure(1)
plt.title("pos vs time")
labels = []

for i in range(3):
    plt.plot(opt.tau, x[1 + i, :], label="")
plt.legend(["x", "y", "z"])
plt.xlabel("time (s)")
plt.ylabel("position (m)")

plt.figure(2)
plt.title("u")
labels = []

for i in range(3):
    plt.plot(opt.tau, u[i, :], label="")
plt.legend(["ux", "uy", "uz"])
plt.xlabel("time (s)")
plt.ylabel("thrust vector (m)")

plt.figure(3)
plt.title("virtual control")
labels = []

for i in range(nu.shape[0]):
    plt.plot(opt.tau[1:], nu[i, :], label="")
    print(np.linalg.norm(max(nu[i,:])))
    
    plt.xlabel("time (s)")
    plt.ylabel("virtual control")

# 3d trajectory plot
fig_traj_plot = plt.figure(4, figsize=(8, 8))
# fig_traj_plot.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
fig_traj_plot.tight_layout()
ax = plt.axes(projection="3d")
ax.view_init(elev=15, azim=-160)
ax.plot3D(x[2, :], x[3, :], x[1, :])


# fix aspect ratio of 3d plot
x_lim = ax.get_xlim3d()
y_lim = ax.get_ylim3d()
z_lim = ax.get_zlim3d()

max_lim = max(
    abs(x_lim[1] - x_lim[0]), abs(y_lim[1] - y_lim[0]), abs(z_lim[1] - z_lim[0])
)
x_mid = sum(x_lim) * 0.5
y_mid = sum(y_lim) * 0.5

rt_I = np.zeros((3, opt.nk))
for i in range(opt.nk):
    DCM = np.array(
        [
            [
                1 - 2 * (x[7 + 2, i] ** 2 + x[7 + 3, i] ** 2),
                2 * (x[7 + 1, i] * x[7 + 2, i] + x[7 + 0, i] * x[7 + 3, i]),
                2 * (x[7 + 1, i] * x[7 + 3, i] - x[7 + 0, i] * x[7 + 2, i]),
            ],
            [
                2 * (x[7 + 1, i] * x[7 + 2, i] - x[7 + 0, i] * x[7 + 3, i]),
                1 - 2 * (x[7 + 1, i] ** 2 + x[7 + 3, i] ** 2),
                2 * (x[7 + 2, i] * x[7 + 3, i] + x[7 + 0, i] * x[7 + 1, i]),
            ],
            [
                2 * (x[7 + 1, i] * x[7 + 3, i] + x[7 + 0, i] * x[7 + 2, i]),
                2 * (x[7 + 2, i] * x[7 + 3, i] - x[7 + 0, i] * x[7 + 1, i]),
                1 - 2 * (x[7 + 1, i] ** 2 + x[7 + 2, i] ** 2),
            ],
        ]
    )

    rt_I[:, [i]] = 10 * DCM.T @ opt.rt

thrust_vecs = np.empty((3, opt.nk))
qlen = 0.05 * max_lim

for i in range(rt_I.shape[1]):
    thrust_vecs[:, [i]] = DCM.T @ u[:, [i]]

q = qlen * thrust_vecs

base_x = x[1, :] + rt_I[0, :] - q[0, :]
base_y = x[2, :] + rt_I[1, :] - q[1, :]
base_z = x[3, :] + rt_I[2, :] - q[2, :]

ax.quiver(
    base_y,
    base_z,
    base_x,
    q[1, :],
    q[2, :],
    q[0, :],
    normalize=False,
    arrow_length_ratio=0.1,
    color="red",
    linewidth=0.5,
)

base_x_2 = x[1, :] + rt_I[0, :]
base_y_2 = x[2, :] + rt_I[1, :]
base_z_2 = x[3, :] + rt_I[2, :]

ax.quiver(
    base_y_2,
    base_z_2,
    base_x_2,
    -2 * rt_I[1, :],
    -2 * rt_I[2, :],
    -2 * rt_I[0, :],
    normalize=False,
    arrow_length_ratio=0.1,
    color="black",
    linewidth=1.0,
)

ax.set_xlim3d([x_mid - max_lim * 0.5, x_mid + max_lim * 0.5])
ax.set_ylim3d([y_mid - max_lim * 0.5, y_mid + max_lim * 0.5])
ax.set_zlim3d([0, max_lim])
ax.plot(ax.get_xlim(), (0, 0), (0, 0), color="black", linestyle="--", linewidth=1)
ax.plot((0, 0), ax.get_ylim(), (0, 0), color="black", linestyle="--", linewidth=1)


def shared_traj_plot_properties(ax):
    ax.set_title("fuel optimal trajectory")
    ax.scatter(0, 0, 0, color="green", s=10)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_zlabel("x")


shared_traj_plot_properties(ax)

############################# animation #############################

fig_anim = plt.figure(5, figsize=(8, 8))
fig_anim.tight_layout()
ax_anim = plt.axes(projection="3d")
ax_anim.view_init(elev=15, azim=-160)
ax_anim.plot3D(x[2, :], x[3, :], x[1, :], linestyle="--", linewidth=0.5, color="black")
shared_traj_plot_properties(ax_anim)
ax_anim.set_xlim(ax.get_xlim())
ax_anim.set_ylim(ax.get_ylim())
ax_anim.set_zlim(ax.get_zlim())

quiver = ax_anim.quiver(
    base_y[0],
    base_z[0],
    base_x[0],
    q[1, 0],
    q[2, 0],
    q[0, 0],
    normalize=False,
    arrow_length_ratio=0.1,
    color="red",
    linewidth=1,
)
quiver2 = ax_anim.quiver(
    base_y_2[0],
    base_z_2[0],
    base_x_2[0],
    -2 * rt_I[1, 0],
    -2 * rt_I[2, 0],
    -2 * rt_I[0, 0],
    normalize=False,
    arrow_length_ratio=0.1,
    color="black",
    linewidth=1,
)


def update(frame):
    quiver.set_segments(
        [
            [
                [base_y[frame], base_z[frame], base_x[frame]],
                [
                    x[2, frame] + rt_I[1, frame],
                    x[3, frame] + rt_I[2, frame],
                    x[1, frame] + rt_I[0, frame],
                ],
            ]
        ]
    )

    quiver2.set_segments(
        [
            [
                [base_y_2[frame], base_z_2[frame], base_x_2[frame]],
                [
                    base_y_2[frame] - 2 * rt_I[1, frame],
                    base_z_2[frame] - 2 * rt_I[2, frame],
                    base_x_2[frame] - 2 * rt_I[0, frame],
                ],
            ]
        ]
    )

    return quiver, quiver2


anim_int = 100
animation = FuncAnimation(fig_anim, update, frames=opt.nk, interval=anim_int)

fig_names = ["position", "trajectory", "throttle", "thrusts", "mass", "cost_tof"]

plt.show(block=False)
plt.pause(1)
input()
plt.close()
