from ns_solvers import grid
import numpy as np
from ns_solvers.boundary_conditions import BoundaryCondition, EdgeBoundaryConditions
from ns_solvers.fd_solvers.chorins_method import (
    intermediate_velocity,
    apply_bcs,
    compute_dx_dx2,
    pressure_poisson,
    jacobi
)


def bc1():
    return EdgeBoundaryConditions(
        BoundaryCondition.from_str("dirichlet", (0.0, None), 5.0),
        BoundaryCondition.from_str("neumann", (1.0, None), 0.0),
        BoundaryCondition.from_str("dirichlet", (None, 1.0), 0.0),
        BoundaryCondition.from_str("dirichlet", (None, 2.0), 0.0),
    )


def plot_check(ugrid: grid.UniformStaggeredGrid, u):
    fig = ugrid.plot_vertices(u=u)
    fig.show()


def main():
    ugrid = grid.UniformStaggeredGrid((10, 10), ((0.0, 1.0), (1.0, 2.0)))
    p = np.ones((10, 10), dtype=float).flatten()
    bcs = bc1()
    u: np.ndarray[float] = np.concatenate(  # type: ignore
            [
                np.zeros(ugrid.main_grid.edge_vertices.shape[1] // 2),
                np.zeros(ugrid.main_grid.edge_vertices.shape[1] // 2),
            ]
    )
    u = apply_bcs(u, ugrid.main_grid, bcs)
    pressure_poisson(u, p, ugrid.main_grid, .1, .5)
    b, xi = np.random.rand(100), np.random.rand(100)
    A = np.eye(100) * 2
    x_new = jacobi(A, xi, b, 100)
    print('d')
    # plot_check(ugrid)


if __name__ == "__main__":
    main()
