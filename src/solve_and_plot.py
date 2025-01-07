import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx
from matplotlib.animation import FuncAnimation
from opt_problem import optProblem


opt = optProblem()

def DCM_output(q): 
        return np.array(
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

for i in range(0, 12):
    opt.discretize()
    x, u, nu = opt.solve_cvx_problem()

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

# control effort plots
plt.figure(3)
plt.title("norm(thrust_vector) vs time")
plt.plot(opt.tau[:], np.linalg.norm(u[:, :], axis=0))
plt.xlabel("time (s)")
plt.ylabel("thrust (N)")

plt.figure(4)
plt.title("virtual control")
labels = []

for i in range(nu.shape[0]):
    plt.plot(opt.tau[1:], nu[i, :])
    #print(np.linalg.norm(max(nu[i, :])))

    plt.xlabel("time (s)")
    plt.ylabel("virtual control")
plt.legend()

# 3d trajectory plot
fig_traj_plot = plt.figure(5, figsize=(8, 8))
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
    rt_I[:, [i]] = 10 * DCM_output(x[7:11, i]).T @ opt.rt

thrust_vecs = np.empty((3, opt.nk))
qlen = 0.01 * max_lim

for i in range(rt_I.shape[1]):
    thrust_vecs[:, [i]] = DCM_output(x[7:11, i]).T @ u[:, [i]]

q = qlen * thrust_vecs

base_x = x[1, :] - q[0, :]
base_y = x[2, :] - q[1, :]
base_z = x[3, :] - q[2, :]

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

base_x_2 = x[1, :]
base_y_2 = x[2, :]
base_z_2 = x[3, :]

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
    ax.set_title("time optimal trajectory")
    ax.scatter(0, 0, 0, color="green", s=10)
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_zlabel("x")


shared_traj_plot_properties(ax)

############################# animation #############################

fig_anim = plt.figure(6, figsize=(8, 8))
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
                    x[2, frame],
                    x[3, frame],
                    x[1, frame],
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