from solve import solveProblem
from plot import Plots

solver = solveProblem()
plotter = Plots()

solver.solve()

plotter.plot(solver.opt, solver, solver.x)

