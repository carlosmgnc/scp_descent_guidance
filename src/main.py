import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# class sim:
#   def __init__(self):
  

class opt_problem:
  def __init__(self):
    
    #vehicle properties
    self.g = -1
    self.mw = 2.0
    self.md = 1.0
    self.Isp = 225
    self.T = 5
    self.rt = np.array([[-0.01],[0],[0]])
    self.Jbvec = np.array([[0.01],[0.01],[0.01]])
    self.Jb = np.diag(self.Jbvec.flatten())
    self.Jbinv = np.diag((1/self.Jbvec).flatten())
    self.alpha = 1.0

    self.nk = 50
    self.K = np.arange(0, self.nk)
    self.dt = 1 / (self.nk-1)
    self.tau = np.linspace(0, 1, self.nk)

    #initial conditions
    ri = np.array([[4], [4], [0]])
    vf = np.array([[-0.1], [0], [0]])
    qf = np.array([[1], [0], [0], [0]])
    wi = np.zeros((3, 1))
    vi = np.array([[-0.5], [0], [0]])
    vf = np.array([[-0.5], [0], [0]])
    g = np.array([[-1], [0], [0]])

    # tfguess = (-vi[0, 0] + np.sqrt(vi[0, 0]**2 - 2*ri[0, 0]*g[0, 0]))/g[0, 0]

    tfguess = 70

    #define initial guess trajectory (wow, i didn't know the guess trajectory could be so simple)
    alpha1_list = (self.nk - self.K) / self.nk 
    alpha2_list = self.K / self.nk

    self.mk = alpha1_list*self.mw + alpha2_list*self.md
    self.rk = alpha1_list*ri
    self.vk = alpha1_list*vi + alpha2_list*vf
    self.qk = np.vstack([np.ones((1, self.nk)), np.zeros((3, self.nk))])
    self.wk = np.zeros((3, self.nk))

    self.xk = np.vstack([self.mk, self.rk, self.vk, self.qk, self.wk])
    self.uk = -self.mk*g

    self.A = np.empty((self.nk-1, 14, 14))
    self.B = np.empty((self.nk-1, 14, 3))
    self.C = np.empty((self.nk-1, 14, 3))
    self.E = np.empty((self.nk-1, 14, 1))
    self.z = np.empty((self.nk-1, 14, 1))
    self.sigma = tfguess

  #compute the LTV model for the guess trajectory by taylor series

  def linearize(self, t):
    k = int(np.floor(t / self.dt))
    tk = self.K[k]
    tk1 = self.K[k+1]

    alphak = (tk1 - t) / (tk1 - tk)
    betak =  (t - tk) / (tk1 - tk)

    #interpolated trajectory point
    
    x = alphak * self.xk[:, k] + betak * self.xk[:, k+1]
    m = x[0]
    r = x[1:4]
    v = x[4:7]
    q = x[7:11]
    w = x[11:14]

    u = alphak * self.uk[:,k] + betak * self.uk[:,k+1]

    C = np.array([[1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] + q[0]*q[3]), 2*(q[1]*q[3] - q[0]*q[2])],
                  [2*(q[1]*q[2] - q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] + q[0]*q[1])],
                  [2*(q[1]*q[3] + q[0]*q[2]), 2*(q[2]*q[3] - q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)]])

    omega = np.array([[0, -w[0], -w[1], -w[2]],
                [w[0], 0, w[2], -w[1]],
                [w[1], -w[2], 0, w[0]],
                [w[2], w[1], -w[0], 0]])
    
    rt_cr = np.array([[0, -self.rt[2, 0], self.rt[1, 0]],
                      [self.rt[2, 0], 0, -self.rt[0, 0]],
                      [-self.rt[1, 0], self.rt[0, 0], 0]])
    
    w_cr = np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])
    
    
    # jacobian w.r.t state
    #respective jacobians obtained with sympy
    D_vq = np.array([[-2*q[2]*u[2] + 2*q[3]*u[1], 2*q[2]*u[1] + 2*q[3]*u[2], -2*q[0]*u[2] + 2*q[1]*u[1] - 4*q[2]*u[0], 2*q[0]*u[1] + 2*q[1]*u[2] - 4*q[3]*u[0]],
                     [2*q[1]*u[2] - 2*q[3]*u[0], 2*q[0]*u[2] - 4*q[1]*u[1] + 2*q[2]*u[0], 2*q[1]*u[0] + 2*q[3]*u[2], -2*q[0]*u[0] + 2*q[2]*u[2] - 4*q[3]*u[1]],
                     [-2*q[1]*u[1] + 2*q[2]*u[0], -2*q[0]*u[1] - 4*q[1]*u[2] + 2*q[3]*u[0], 2*q[0]*u[0] - 4*q[2]*u[2] + 2*q[3]*u[1], 2*q[1]*u[0] + 2*q[2]*u[1]]])
    
    D_vq = (1/m)*D_vq
    
    D_qw = np.array([[-0.5*q[1], -0.5*q[2], -0.5*q[3]],
                     [0.5*q[0], -0.5*q[3], 0.5*q[2]],
                     [0.5*q[3], 0.5*q[0], -0.5*q[1]],
                     [-0.5*q[2], 0.5*q[1], 0.5*q[0]]])
    
    D_ww = np.array([[0, (self.Jbvec[1, 0]*w[2] - self.Jbvec[2, 0]*w[2])/self.Jbvec[0, 0], (self.Jbvec[1, 0]*w[1] - self.Jbvec[2, 0]*w[1])/self.Jbvec[0, 0]],
                     [(-self.Jbvec[0, 0]*w[2] + self.Jbvec[2, 0]*w[2])/self.Jbvec[1, 0], 0, (-self.Jbvec[0, 0]*w[0] + self.Jbvec[2, 0]*w[0])/self.Jbvec[1, 0]],
                     [(self.Jbvec[0, 0]*w[1] - self.Jbvec[1, 0]*w[1])/self.Jbvec[2, 0], (self.Jbvec[0, 0]*w[0] - self.Jbvec[1, 0]*w[0])/self.Jbvec[2, 0], 0]])
    u = u.reshape(3, 1)
    w = w.reshape(3, 1)

    D_vm = -(C @ u)/(m**2)
    D_qq = 0.5*omega
    
    D_x1 = np.zeros((1, 14))
    D_x2 = np.hstack([np.zeros((3, 4)), np.eye(3), np.zeros((3, 7))])
    D_x3 = np.hstack([D_vm, np.zeros((3, 6)), D_vq, np.zeros((3, 3))])
    D_x4 = np.hstack([np.zeros((4, 7)), D_qq, D_qw])
    D_x5 = np.hstack([np.zeros((3, 11)), D_ww])
    D_x = np.vstack([D_x1, D_x2, D_x3, D_x4, D_x5])

    # jacobian w.r.t input
    D_u = np.vstack([-np.transpose(u)*self.alpha/np.linalg.norm(u), np.zeros((3, 3)), C/m, np.zeros((4, 3)), self.Jbinv*rt_cr])

    Ac = self.sigma * D_x

    Bc = self.sigma * D_u

    #nonlinear propagation term
    f1 = np.array([[-self.alpha * np.linalg.norm(u)]])
    f2 = v
    f3 = (1/m)*C @ u + self.g
    f4 = 0.5*omega @ q
    f5 = self.Jbinv @ (rt_cr @ u - w_cr @ self.Jb @ w)

    f2 = f2.reshape((3, 1))
    f4 = f4.reshape((4, 1))
    Ec = np.vstack([f1, f2, f3, f4, f5])
    
    x = x.reshape((14, 1))
    zc = -Ac @ x - Bc @ u

    return Ac, Bc, Ec, zc

  def rk4(self, tk, dt_rk, i, j):
    Ac_k, Bc_k, Ec_k, zc_k = self.linearize(tk)
    Ac_k_half, Bc_k_half, Ec_k_half, zc_k_half = self.linearize(tk + dt_rk/2)
    Ac_k_34, _, _, _ = self.linearize(tk + dt_rk*3/4)
    Ac_k_1, Bc_k_1, Ec_k_1, zc_k_1 = self.linearize(tk + dt_rk)

    I14 = np.eye(14)

    if j == 0:
      A0 = I14
      B0 = np.zeros((14, 3))
      C0 = np.zeros((14, 3))
      E0 = np.zeros((14, 1))
      z0 = np.zeros((14, 1))
    else:
      A0 = self.A[i,:,:]
      B0 = self.B[i,:,:]
      C0 = self.C[i,:,:]
      E0 = self.E[i,:,0]
      z0 = self.z[i,:,0]

    #stm tk -> tk+1
    k1 = Ac_k
    k2 = Ac_k_half @ (I14 + (dt_rk/2)*k1)
    k3 = Ac_k_half @ (I14 + (dt_rk/2)*k2)
    k4 = Ac_k_1 @ (I14 + (dt_rk)*k3)

    Ark = A0 + (dt_rk/6) * (k1 + 2*k2 + 2*k3 + k4)
      
    #stm tk + 1/2 -> tk + 1
    k1 = Ac_k_half
    k2 = Ac_k_34 @ (I14 + (dt_rk*3/4)*k1)
    k3 = Ac_k_34 @ (I14 + (dt_rk*3/4)*k2)
    k4 = Ac_k_1 @ (I14 + (dt_rk)*k3)

    stm_half = I14 + (dt_rk/6) * (k1 + 2*k2 + 2*k3 + k4)

    #B, C, E, z integrals
    k1 = Ark @ Bc_k * 1.0
    k2 = stm_half @ Bc_k_half * 0.5
    k3 = stm_half @ Bc_k_half * 0.5
    k4 = 0

    Brk = B0 + (dt_rk/6) * (k1 + 2*k2 + 2*k3 + k4)

    k1 = 0
    k2 = stm_half @ Bc_k_half * 0.5
    k3 = stm_half @ Bc_k_half * 0.5
    k4 = I14 @ Bc_k_1 * 1.0
    
    Crk = C0 + (dt_rk/6) * (k1 + 2*k2 + 2*k3 + k4)

    k1_E = Ark @ Ec_k
    k2_E = stm_half @ Ec_k_half
    k3_E = stm_half @ Ec_k_half
    k4_E = I14 @ Ec_k_1
    
    Erk = E0.reshape((14, 1)) + (dt_rk/6) * (k1_E + 2*k2_E + 2*k3_E + k4_E)

    k1_z = Ark @ zc_k
    k2_z = stm_half @ zc_k_half
    k3_z = stm_half @ zc_k_half
    k4_z = I14 @ zc_k_1
    
    zrk = z0.reshape((14, 1)) + (dt_rk/6) * (k1_z + 2*k2_z + 2*k3_z + k4_z)

    self.A[i,:,:] = Ark 
    self.B[i,:,:] = Brk
    self.C[i,:,:] = Crk
    self.E[i,:,0] = Erk.flatten()
    self.z[i,:,0] = zrk.flatten()

  def discretize(self):
    #RK4 integration between timesteps with FOH
    nsub = 5
    dt_sub = self.dt/(nsub + 1)
     
    for i in range(0, self.A.shape[0]):
      #rk4 integration
      for j in range(0, nsub-1):
          sub_time = i*self.dt + j*dt_sub
          self.rk4(sub_time, dt_sub, i, j)

    # array = self.A[48,:, :]
    # formatted_array = np.array([[f"{x:.4g}" for x in row] for row in array])
    # print("\n".join(["\t".join(row) for row in formatted_array]))

  def solve_cvx_problem(self, A, B, C, E, z):
    x = cvx.variable((14, self.nk))
    u = cvx.variable((3, self.nk-1))

    cost = 0
    constraints = []
    #boundary constraints
    constraints += [x[0,0] == self.mw]
    # constraints += [x[1:4,0] == ]

    #dynamics equality constraints

    #state inequality constraints

    #control constraints

    objective = cvx.Minimize(cost)
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    print("solver status: " + prob.status)
    if prob.status != 'optimal':
      opt_cost = float('inf')
      return opt_cost, None, None
    
    else:
      opt_cost = cost.value
      traj_Nt = x.value
      u_Nt = u.value
      return opt_cost, x.value, u.value

    return x, u, sigma
  
  #def iterate():

opt = opt_problem()

opt.discretize()
#for i in range(0, 15):
  #opt.discretize()
  # opt.solve_cvx_problem()

