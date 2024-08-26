from ns_solvers import grid
import numpy as np
import plotly.graph_objs as go
from ns_solvers.boundary_conditions import BoundaryCondition, EdgeBoundaryConditions
from ns_solvers.fd_solvers.chorins_method import chorin_step


def bc1():
    return EdgeBoundaryConditions(
        BoundaryCondition.from_str("dirichlet", (0.0, None), 1.0),
        BoundaryCondition.from_str("dirichlet", (1.0, None), 1.0),
        BoundaryCondition.from_str("dirichlet", (None, 1.0), 0.0),
        BoundaryCondition.from_str("dirichlet", (None, 2.0), 0.0),
    )


def plot_check(ugrid: grid.StaggeredGrid):
    ugrid.plot_cells()


def test_case_1(ugrid: grid.StaggeredGrid):
    p = np.ones(ugrid.n_cells).flatten()
    u = np.zeros(ugrid.edges.shape[1])
    rho = 0.1
    nu = 0.1
    dt = 0.1
    bc = bc1()
    for i in range(5):
        u, p = chorin_step(u, p, ugrid, dt, rho, nu, bc)  # type: ignore
    p = p.reshape(ugrid.n_cells)
    fig = go.Figure(go.Heatmap(z=p))
    fig.show()


def main():
    ugrid = grid.StaggeredGrid((10, 10), ((0.0, 1.0), (1.0, 2.0)))
    plot_check(ugrid)
    test_case_1(ugrid)


if __name__ == "__main__":
    main()
