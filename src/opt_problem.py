import numpy as np
import cvxpy as cvx
import sympy as sy
from sympy import *

class optProblem:
    def __init__(self):

        # vehicle properties
        self.mw = 2
        self.md = 1
        self.Tmax = 5
        self.Tmin = 0.3
        self.rt = np.array([[-0.01], [0], [0]])
        self.Jbvec = np.array([[0.01], [0.01], [0.01]])
        self.Jb = np.diag(self.Jbvec.flatten())
        self.Jbinv = np.diag((1 / self.Jbvec).flatten())
        self.alpha = 0.01

        #discrete time grid
        self.nk = 50
        self.K = np.arange(0, self.nk)
        self.dt = 1 / (self.nk - 1)
        self.tau = np.linspace(0, 1, self.nk)

        # initial conditions
        ri = np.array([[4], [4], [0]])
        vi = np.array([[-0.5], [-2], [2]])
        vf = np.array([[0], [-0.1], [0]])
        self.g = np.array([[-1], [0], [0]])

        tfguess = 5
        alpha1_list = (1 - self.tau) / (1)
        alpha2_list = 1 - alpha1_list

        self.mk = alpha1_list * self.mw + alpha2_list * self.md
        self.rk = alpha1_list * ri
        self.vk = alpha1_list * vi + alpha2_list * vf
        self.qk = np.vstack([np.ones((1, self.nk)), np.zeros((3, self.nk))])
        self.wk = np.zeros((3, self.nk))

        self.xk = np.vstack([self.mk, self.rk, self.vk, self.qk, self.wk])
        self.uk = -self.mk * self.g
        self.sigmak = tfguess

        self.A = np.empty((self.nk - 1, 14, 14))
        self.B = np.empty((self.nk - 1, 14, 3))
        self.C = np.empty((self.nk - 1, 14, 3))
        self.E = np.empty((14, self.nk - 1))
        self.z = np.empty((14, self.nk - 1))

        self.stm_inv = np.zeros((14, 14))
        self.A_f, self.B_f, self.E_f, self.z_f = self.linearize()

    # compute the LTV model for the guess trajectory by taylor series
    def DCM(self, q): 
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
    
    def omega(self, w):
        return sy.Matrix(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ]
    )

    def cr(self, v):
        return sy.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    def get_alphas(self, t):
        k = int(np.floor(t / self.dt))
        tk = self.tau[k]

        # FOH interpolation of control input
        if t != 1:
            tk1 = self.tau[k + 1]
            alphak = (tk1 - t) / (tk1 - tk)
            betak = 1 - alphak
        else:
            tk1 = 1
            alphak = 1
            betak = 0

        return alphak, betak, tk, tk1, k
    

    def linearize(self):
        f_symb = sy.zeros(14, 1)
        x = sy.Matrix(sy.symbols("m r0 r1 r2 v0 v1 v2 q0 q1 q2 q3 w0 w1 w2", real=True))
        u = sy.Matrix(sy.symbols("u0 u1 u2", real=True))

        g = sy.Matrix(self.g)
        rt = sy.Matrix(self.rt)
        Jb = sy.Matrix(self.Jb)
        Jbinv = sy.Matrix(self.Jbinv)

        f_symb[0, 0] = -self.alpha * u.norm()
        f_symb[1:4, 0] = x[4:7, 0]
        f_symb[4:7, 0] = (1/x[0, 0]) * self.DCM(x[7:11, 0]).T * u + g
        f_symb[7:11, 0] = (1/2) * self.omega(x[11:14, 0]) * x[7:11, 0]
        f_symb[11:14, 0] = Jbinv * (self.cr(rt) * u - self.cr(x[11:14, 0]) * Jb * x[11:14, 0])

        f_symb = f_symb
        A_symb = f_symb.jacobian(x) * self.sigmak
        B_symb = f_symb.jacobian(u) * self.sigmak
        z_symb = - A_symb * x - B_symb * u

        A_f = sy.lambdify((x, u), A_symb, 'numpy')
        B_f = sy.lambdify((x, u), B_symb, 'numpy')
        E_f = sy.lambdify((x, u), f_symb, 'numpy')
        z_f = sy.lambdify((x, u), z_symb, 'numpy')

        return A_f, B_f, E_f, z_f

    # derivative function for rk4
    def P_dot(self, t, X):
        phi = X[14 : 14 + 14 * 14].reshape((14, 14))
        phi_inv = np.linalg.inv(phi)

        alphak, betak, tk, tk1, k = self.get_alphas(t)
        u = (alphak * self.uk[:, k] + betak * self.uk[:, k]).reshape((3, 1))

        X_flat = X[:14].flatten()
        u_flat = u.flatten()

        P1 = self.E_f(X_flat, u_flat)
        P2 = (self.A_f(X_flat, u_flat) @ phi).reshape((14 * 14, 1))
        P3 = (phi_inv @ self.B_f(X_flat, u_flat) * alphak).reshape((14 * 3, 1))
        P4 = (phi_inv @ self.B_f(X_flat, u_flat) * betak).reshape((14 * 3, 1))
        P5 = phi_inv @ self.E_f(X_flat, u_flat)
        P6 = phi_inv @ self.z_f(X_flat, u_flat)

        return np.vstack([P1, P2, P3, P4, P5, P6])

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

        #indeces for flattened state
        nsq = 14 * 14

        xnd = 14
        stmnd = 14 + nsq
        Bnd = 4 * 14 + nsq
        Cnd = 7 * 14 + nsq
        End = 8 * 14 + nsq
        znd = 9 * 14 + nsq

        P_temp = np.empty((znd, 1))

        P_temp[:xnd, [0]] = self.xk[:, [0]]
        P_temp[xnd : stmnd, [0]] = np.eye(14).reshape((nsq, 1))
        P_temp[stmnd : Bnd, [0]] = np.zeros((14, 3)).reshape((14 * 3, 1))
        P_temp[Bnd: Cnd, [0]] = np.zeros((14, 3)).reshape((14 * 3, 1))
        P_temp[Cnd : End, [0]] = np.zeros((14, 1))
        P_temp[End : znd, [0]] = np.zeros((14, 1))

        for i in range(0, self.nk - 1):
            for j in range(0, nsub + 1):
                sub_time = i * self.dt + j * dt_sub
                P_temp = self.rk41(self.P_dot, sub_time, P_temp, dt_sub)

            stm = P_temp[xnd : stmnd].reshape((14, 14))
            self.A[i, :, :] = stm
            self.B[i, :, :] = stm @ P_temp[stmnd : Bnd].reshape((14, 3))
            self.C[i, :, :] = stm @ P_temp[Bnd : Cnd].reshape((14, 3))
            self.E[:, [i]] = stm @ P_temp[Cnd : End].reshape((14, 1))
            self.z[:, [i]] = stm @ P_temp[End : znd].reshape((14, 1))

            P_temp[:xnd, [0]] = self.xk[:, [i]]
            P_temp[xnd: stmnd, [0]] = np.eye(14).reshape((nsq, 1))
            P_temp[stmnd : Bnd, [0]] = np.zeros((14, 3)).reshape((14 * 3, 1))
            P_temp[Bnd: Cnd, [0]] = np.zeros((14, 3)).reshape((14 * 3, 1))
            P_temp[Cnd: End, [0]] = np.zeros((14, 1))
            P_temp[End : znd, [0]] = np.zeros((14, 1))

    def solve_cvx_problem(self):
        x = cvx.Variable((14, self.nk))
        u = cvx.Variable((3, self.nk))
        sigma = cvx.Variable((1, 1), nonneg=True)
        nu = cvx.Variable((14, self.nk - 1))

        w_nu = 100000 

        theta_max = np.deg2rad(90)
        deltamax = np.deg2rad(20)
        w_max = np.deg2rad(60)
        gamma_gs = np.deg2rad(20)

        nu_cost = 0
        dz_cost = 0

        W = np.diag([100 for x in range(18)])

        constraints = []
        constraints += [self.md <= x[0, :]]
        constraints += [sigma >= 1]

        for k in range(self.nk):
            # soft trust region cost
            dx = x[:,[k]]-self.xk[:, [k]]
            du = u[:, [k]]-self.uk[:, [k]]
            dsigma = sigma - self.sigmak

            dz = cvx.vstack([dx, du, dsigma])
            dz_cost += cvx.quad_form(dz, W)

            #state and control constraints
            H = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
            constraints += [np.cos(theta_max) <= 1 - 2 * cvx.square(cvx.norm(H @ x[7:11, [k]]))]
            constraints += [cvx.norm(x[11:14, [k]]) <= w_max]

            constraints += [cvx.norm(u[:, [k]]) <= self.Tmax]

            constraints += [
                self.Tmin
                <= (np.transpose(self.uk[:, [k]]) / np.linalg.norm(self.uk[:, [k]]))
                @ u[:, [k]]
            ]

            constraints += [np.cos(deltamax) * cvx.norm(u[:, [k]]) <= u[0, [k]]]

            e2 = np.array([[0], [1], [0]])
            e3 = np.array([[0], [0], [1]])
            H23 = np.hstack([e2, e3])

            constraints += [np.tan(gamma_gs) * cvx.norm(H23.T @ x[1:4, [k]]) <= x[1, [k]]]
        
        #boundary conditions
        constraints += [x[0, 0] == self.xk[0, 0]]
        constraints += [x[1:4, 0] == self.xk[1:4, 0]]
        constraints += [x[4:7, 0] == self.xk[4:7, 0]]
        constraints += [x[11:14, 0] == self.xk[11:14, 0]]

        #constraints += [x[1:, 0] == self.xk[1:, 0]]
        constraints += [x[1:, -1] == self.xk[1:, -1]]

        #stay above ground
        constraints += [x[1, :] >= 0]

        #dynamics constrains
        for k in range(0, self.nk - 1):
            constraints += [
                x[:, [k + 1]]
                == self.A[k, :, :] @ x[:, [k]]
                + self.B[k, :, :] @ u[:, [k]]
                + self.C[k, :, :] @ u[:, [k + 1]]
                + self.E[:, [k]] * sigma[0, 0]
                + self.z[:, [k]]
                + nu[:, [k]]
            ]

            nu_cost += cvx.norm(nu[:, [k]], 1)
        
        #total cost
        cost = (
            sigma
            + dz_cost
            + w_nu * nu_cost
        )

        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective, constraints)

        print("----------------------------------")
        print("solving")
        prob.solve("CLARABEL")
        print("solver status : " + prob.status)
        print("solve time    :" + str(prob.solver_stats.solve_time))
        print("cost          :" + str(prob.objective.value))
        print("sigma         : " + str(sigma.value))
        print("----------------------------------")

        self.xk = x.value
        self.uk = u.value
        self.sigmak = sigma.value

        return x.value, u.value, sigma.value, cost, nu.value