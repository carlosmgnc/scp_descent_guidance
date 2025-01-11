import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sy
from sympy import *
from plot import Plots
from solve import solveProblem

from opt_problem import optProblem

class simulation:
    def __init__(self, opt, solver):
        
        self.opt = opt
        self.solver = solver

        self.trajectory = np.zeros((14, self.opt.nk))
        self.trajectory[:, [0]] = self.solver.x[:, [0]]

        self.u = self.solver.u
        self.sigma_opt = self.solver.sigma

    def nonlin_func(self, t, X):
        x = X.flatten()
        alphak, betak, tk, tk1, k = self.opt.get_alphas(t)
        u = (alphak * self.u[:, k] + betak * self.u[:, k]).reshape((3, 1))
        u = u.flatten()

        def sigmaE_f(E_f):
            
            return self.sigma_opt * E_f


        return sigmaE_f(self.opt.E_f(x, u))
        
    def integrate_full_trajectory(self):

        nsub = 15
        dt_sub = self.opt.dt / (nsub + 1)


        P_temp = self.trajectory[:, [0]]

        for i in range(0, self.opt.nk - 1):
            for j in range(0, nsub + 1):
                sub_time = i * self.opt.dt + j * dt_sub
                P_temp = self.opt.rk41(self.nonlin_func, sub_time, P_temp, dt_sub)

            self.trajectory[:, [i + 1]] = P_temp

solver = solveProblem()
plotter = Plots()
solver.solve()

sim = simulation(solver.opt, solver)
sim.integrate_full_trajectory()

print("final position: " + str(sim.trajectory[1:4, -1]))
plotter.plot(sim.opt, solver, sim.trajectory)

