"""
Chorins Method on staggered grid
"""

import numpy as np
from ns_solvers.grid import StaggeredGrid
from ns_solvers.boundary_conditions import EdgeBoundaryConditions, BCType


def apply_bcs(
    u: np.ndarray[float], grid: StaggeredGrid, bcs: EdgeBoundaryConditions
) -> np.ndarray[float]:
    # Apply all bcs to u
    for idx, (name, bc) in enumerate(bcs.all_bcs()):
        mask = bc.applies_mask(grid.edges)
        if bc.bc_type == BCType.DIRICHLET:
            u[mask] = bc.value
        elif bc.bc_type == BCType.NEUMANN:
            neighbor_direction = 0 if name in ["right", "top"] else 1
            # In theory all neighbors should not be -1 for internal edges
            u[mask] = u[grid.edge_neighbors[neighbor_direction][mask]]  # du/dx = 0

    return u


def compute_dx_dx2(
    u: np.ndarray[float], grid: StaggeredGrid
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """
    Compute the gradient of u
    Assuming the edge points are handled before this function
    This function will compute gradient and laplacian for all points inside of U
    if u is M x N, then the function will return M-2 x N-2
    :param u: velocity field
    :param grid: grid object
    :return:
    """
    u_interior = u[grid.internal_edge_mask]
    l_neighbors = grid.edge_neighbors[0, grid.internal_edge_mask]
    r_neighbors = grid.edge_neighbors[1, grid.internal_edge_mask]
    dx = np.array(grid.dx)[grid.edge_directions][grid.internal_edge_mask]
    du_dx = (u[r_neighbors] - u[l_neighbors]) / dx
    d2u_dx2 = (u[r_neighbors] - 2 * u_interior + u[l_neighbors]) / dx**2
    return du_dx, d2u_dx2


def intermediate_velocity(
    u: np.ndarray[float], grid: StaggeredGrid, dt: float, nu: float
) -> np.ndarray[float]:
    """
    Calculate intermediate velocity field
    :return:
    """
    du_dx, d2u_dx2 = compute_dx_dx2(u, grid)
    u_internal = u[grid.internal_edge_mask]
    u[grid.internal_edge_mask] = u_internal + dt * (-du_dx * u_internal + nu * d2u_dx2)
    return u


def jacobi(
    A: np.ndarray[float],
    xi: np.ndarray[float],
    b: np.ndarray[float],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray[float]:
    D = np.diag(A)
    R = A - np.diagflat(D)

    for i in range(max_iter):
        xt = (b - np.dot(R, xi)) / D
        if np.linalg.norm(xt - xi) < tol:
            return xt
        xi = xt

    print(f"WARNING Did not Converge after {max_iter} iterations")
    return xi


def pressure_poisson(
    u: np.ndarray[float],
    p: np.ndarray[float],
    grid: StaggeredGrid,
    dt: float,
    rho: float,
) -> np.ndarray[float]:
    """
    Solve the pressure poisson equation
    :param u: velocity field
    :param grid: grid object
    :param dt: time step
    :param rho: density
    :return:
    """
    du_dx1 = (
        u[grid.center_to_edge_neighbors["right"]]
        - u[grid.center_to_edge_neighbors["left"]]
    ) / grid.dx[0]
    du_dx2 = (
        u[grid.center_to_edge_neighbors["top"]]
        - u[grid.center_to_edge_neighbors["bottom"]]
    ) / grid.dx[1]
    domain_du_dx = du_dx1 + du_dx2

    du_dx = np.zeros(grid.n_cells).flatten()
    du_dx[~grid.ghost_node_mask] = domain_du_dx

    p_new = jacobi(grid.laplacian, p, rho / dt * du_dx, max_iter=10000)
    return p_new


def p_grad(p: np.ndarray[float], grid: StaggeredGrid) -> np.ndarray[float]:
    """
    Compute the gradient of p
    :param p: pressure field
    :param grid: grid object
    :return:
    """
    p = p.reshape(grid.n_cells)
    dp_dx1 = (p[1:-1, 1:] - p[1:-1, :-1]) / grid.dx[0]
    dp_dx2 = (p[1:, 1:-1] - p[:-1, 1:-1]) / grid.dx[1]
    return np.hstack([dp_dx1.flatten(), dp_dx2.flatten()])  # type: ignore


def chorin_step(
    u: np.ndarray[float],
    p: np.ndarray[float],
    grid: StaggeredGrid,
    dt: float,
    rho: float,
    nu: float,
    bcs: EdgeBoundaryConditions,
):
    u = apply_bcs(u, grid, bcs)
    intermediate_u = intermediate_velocity(u, grid, dt, nu)
    p_new = pressure_poisson(intermediate_u, p, grid, dt, rho)
    u_new = intermediate_u * dt / rho * p_grad(p_new, grid)
    return u_new, p_new
