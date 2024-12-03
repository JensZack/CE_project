from ns_solvers import grid
import numpy as np
import plotly.graph_objs as go
from ns_solvers.boundary_conditions import BoundaryCondition, EdgeBoundaryConditions
from ns_solvers.fd_solvers.chorins_method import chorin_step


def bc1():
    return EdgeBoundaryConditions(
        BoundaryCondition.from_str("dirichlet", (0.0, None), 0.0),
        BoundaryCondition.from_str("dirichlet", (1.0, None), 0.0),
        BoundaryCondition.from_str("dirichlet", (None, 1.0), 0.0),
        BoundaryCondition.from_str("dirichlet", (None, 2.0), 0.0),
    )


def plot_check(ugrid: grid.StaggeredGrid):
    ugrid.plot_cells()


def plot_laplacian(ugrid: grid.StaggeredGrid):
    fig = go.Figure(go.Heatmap(z=ugrid.laplacian))
    fig.show()


def test_laplacian(ugrid: grid.StaggeredGrid):
    eigvals = np.linalg.eigvals(ugrid.laplacian)
    absvals = np.abs(eigvals)
    print(f"min eigval: {absvals.min()}, max eigval: {absvals.max()}")
    print(f"Condition number: {absvals.max() / absvals.min()}")
    print(f"Matrix is positive definite {np.all(eigvals > 0)}")

def test_case_1(ugrid: grid.StaggeredGrid):
    p = np.ones(ugrid.n_cells)
    p[20, 20] = 200.0
    p = p.flatten()
    print(f"Initial average pressure: {np.mean(p[~ugrid.ghost_node_mask])}")
    u = np.random.rand(ugrid.edges.shape[1])
    rho = 0.001  # density?
    nu = .0001  # viscosity?
    dt = 0.001
    bc = bc1()
    for i in range(10):
        u, p = chorin_step(u, p, ugrid, dt, rho, nu, bc)  # type: ignore
    print(f"Final average pressure: {np.mean(p[~ugrid.ghost_node_mask])}")
    # p = p.reshape(ugrid.n_cells)
    # fig = go.Figure(go.Heatmap(z=p))
    # fig.show()


def main():
    ugrid = grid.StaggeredGrid((40, 40), ((0.0, 1.0), (1.0, 2.0)))
    test_laplacian(ugrid)
    # plot_laplacian(ugrid)
    # plot_check(ugrid)
    test_case_1(ugrid)


if __name__ == "__main__":
    main()
