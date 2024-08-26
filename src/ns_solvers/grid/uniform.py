"""
Creates a staggered uniform grid for FD navier stokes solvers
"""

import functools
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np
from plotly import graph_objects as go


class CellType(Enum):
    GHOST = auto()
    DOMAIN = auto()


@dataclass
class CellNeighbor:
    left: int | None = field(default=lambda: None)
    right: int | None = field(default=lambda: None)
    bottom: int | None = field(default=lambda: None)
    top: int | None = field(default=lambda: None)


class Cell:
    def __init__(self, cell_type: CellType, center: tuple[float, ...]):
        self.cell_type = cell_type
        self.center = center
        self.edge_idxs = CellNeighbor()
        self.cell_neighbors = CellNeighbor()

    def __repr__(self):
        return f"Cell({self.cell_type=})"


class StaggeredGrid:
    """
    With an input of n_cells = (8, 3) and ghost_nodes = True
    creates a staggered grid with the following structure:
    *   *   *   *   *   *   *   *   *   *
    -   -   -   -   -   -   -   -   -   -
    * | x | x | x | x | x | x | x | x | *
    -   -   -   -   -   -   -   -   -   -
    * | x | x | x | x | x | x | x | x | *
    -   -   -   -   -   -   -   -   -   -
    * | x | x | x | x | x | x | x | x | *
    -   -   -   -   -   -   -   -   -   -
    *   *   *   *   *   *   *   *   *   *

    * : ghost node
    x : domain node
    | : x-edge
    - : y-edge

    And defines important properties such as edge_neighbors, laplacian, and other methods
    for running chorins method on a staggered grid
    """

    def __init__(
        self,
        n_cells: tuple[int, ...],
        domain: tuple[tuple[float, float], ...],
        use_ghost_nodes: bool = True,
    ):
        self.use_ghost_nodes = use_ghost_nodes
        self.n_dim = len(n_cells)
        self.domain = domain
        self.n_cells = tuple(c + 2 * int(use_ghost_nodes) for c in n_cells)  # Add ghost
        self.n_domain_cells = n_cells
        self.dx = tuple((d[1] - d[0]) / n for d, n in zip(domain, n_cells))

        self.edges, self.edge_directions = self._create_edges_2d()
        self.cells = self._create_cells_2d()

    def plot_cells(self):
        """
        plot cells, centers and edges
        :return:
        """
        fig = go.Figure()

        # plot centers
        x_d, y_d = zip(
            *[cell.center for cell in self.cells if cell.cell_type == CellType.DOMAIN]
        )
        fig.add_trace(
            go.Scatter(
                x=x_d,
                y=y_d,
                mode="markers",
                name="domain",
                marker=dict(color="blue"),
            )
        )
        if self.use_ghost_nodes:
            x_g, y_g = zip(
                *[
                    cell.center
                    for cell in self.cells
                    if cell.cell_type == CellType.GHOST
                ]
            )
            fig.add_trace(
                go.Scatter(
                    x=x_g,
                    y=y_g,
                    mode="markers",
                    name="ghost",
                    marker=dict(color="blue", symbol="circle-open"),
                )
            )

        # plot mesh lines
        x_edges = np.unique(self.edges[:, self.edge_directions == 0][0])
        y_edges = np.unique(self.edges[:, self.edge_directions == 1][1])
        for x in x_edges:
            fig.add_vline(x, line=dict(color="blue", width=0.5))
        for y in y_edges:
            fig.add_hline(y, line=dict(color="blue", width=0.5))

        fig.show()

    @functools.cached_property
    def internal_edge_mask(self) -> np.ndarray[bool]:
        """
        Get mask for internal edges
        :return:
        """
        edge_masks = []
        # for idx, edge_centers in enumerate(self.edge_vertices):
        for dim in range(self.n_dim):
            edge_centers = self.edges[:, self.edge_directions == dim]
            mask = np.zeros(edge_centers.shape[1], dtype=bool)
            for i in range(self.n_dim):
                mask = np.logical_or(
                    mask, np.isclose(edge_centers[i], self.domain[i][0])
                )
                mask = np.logical_or(
                    mask, np.isclose(edge_centers[i], self.domain[i][1])
                )
            edge_masks.append(~mask)
        return np.hstack(edge_masks)  # type: ignore

    @functools.cached_property
    def edge_neighbors(self):
        """
        Create a 2 x n matrix where the first row is the left neighbor and the second row is the right neighbor
        of each edge in the staggered grid
        :return:
        """
        n_edges = self.edges.shape[1]
        strides = np.concatenate(
            [
                np.ones(np.sum(self.edge_directions == i), dtype=int)
                * self.domain_cell_strides[i]
                for i in range(self.n_dim)
            ]
        )
        l_neighbors = np.arange(n_edges, dtype=int) - strides
        r_neighbors = np.arange(n_edges, dtype=int) + strides

        for i in range(self.n_dim):
            no_l_neighbor = np.logical_and(
                self.edge_directions == i, self.edges[i] == self.domain[i][0]
            )
            no_r_neighbor = np.logical_and(
                self.edge_directions == i, self.edges[i] == self.domain[i][1]
            )
            l_neighbors[no_l_neighbor] = -1
            r_neighbors[no_r_neighbor] = -1

        return np.vstack([l_neighbors, r_neighbors])

    def _create_cells_2d(self) -> list[Cell]:
        """
        Creates a 2d grid, adding ghost nodes if specified in init
        :return:
        """
        # Generate cells left to right and bottom to top
        # Add ghost cells outside of domain
        ops = (np.subtract, np.add) if self.use_ghost_nodes else (np.add, np.subtract)
        centers = np.meshgrid(
            np.linspace(
                ops[0](self.domain[0][0], self.dx[0] / 2),
                ops[1](self.domain[0][1], self.dx[0] / 2),
                self.n_cells[0],
            ),
            np.linspace(
                ops[0](self.domain[1][0], self.dx[1] / 2),
                ops[1](self.domain[1][1], self.dx[1] / 2),
                self.n_cells[1],
            ),
            copy=False,
        )

        if self.use_ghost_nodes:
            # determine ghost cells
            ghost_mask_1 = np.logical_or(
                np.isclose(centers[0], self.domain[0][0] - self.dx[0] / 2),
                np.isclose(centers[0], self.domain[0][1] + self.dx[0] / 2),
            )
            ghost_mask_2 = np.logical_or(
                np.isclose(centers[1], self.domain[1][0] - self.dx[1] / 2),
                np.isclose(centers[1], self.domain[1][1] + self.dx[1] / 2),
            )
            ghost_mask = np.logical_or(ghost_mask_1, ghost_mask_2)

            ghost_iter = (
                CellType.GHOST if g else CellType.DOMAIN for g in ghost_mask.flatten()
            )
        else:
            ghost_iter = (CellType.DOMAIN for _ in range(np.prod(self.n_cells)))

        center_iter = (
            (x, y) for x, y in zip(centers[0].flatten(), centers[1].flatten())
        )

        cells = list(
            Cell(c_type, center) for c_type, center in zip(ghost_iter, center_iter)
        )
        return cells

    def _create_edges_2d(self):
        """
        Given a 2d grid, create the edge vertices and edge directions
        :return:
        """
        edge_vertices = []
        edge_directions = []
        for ind in range(self.n_dim):
            coord_gen = (
                np.linspace(
                    lb + 0.5 * dx * int(idx != ind),
                    ub - 0.5 * dx * int(idx != ind),
                    n + int(idx == ind),
                )
                for idx, ((lb, ub), n, dx) in enumerate(
                    zip(self.domain, self.n_domain_cells, self.dx)
                )
            )
            # Really new edge centers
            new_edges = np.vstack(
                list(map(np.ravel, np.meshgrid(*coord_gen))), dtype=float
            )
            edge_vertices.append(new_edges)
            edge_directions.append(np.ones(new_edges.shape[1], dtype=int) * ind)

        return (  # type: ignore
            np.hstack(edge_vertices),
            np.hstack(edge_directions),
        )

    @functools.cached_property
    def center_to_edge_neighbors(self) -> dict[str, np.ndarray[int]]:
        """
        Create a dictionary of the indices of the edge neighbors for each cell
        :return:
        """
        l_neighbors = np.concatenate(
            [
                np.arange(self.n_domain_cells[0]) + i * (self.n_domain_cells[0] + 1)
                for i in range(self.n_domain_cells[1])
            ]
        )
        x_edges_count = int(np.sum(self.edge_directions == 0))
        r_neighbors = l_neighbors + self.domain_cell_strides[0]
        b_neighbors = np.arange(np.prod(self.n_domain_cells)) + x_edges_count
        t_neighbors = b_neighbors + self.domain_cell_strides[1]
        return {  # type: ignore
            "left": l_neighbors,
            "right": r_neighbors,
            "bottom": b_neighbors,
            "top": t_neighbors,
        }

    @functools.cached_property
    def laplacian(self):
        """
        Create the laplacian operator matrix for cells,
        Currently setting ghost nodes equal to their neighbor domain nodes
        as a boundary condition in the laplacian
        :return:
        """
        n_points = int(np.prod(self.n_cells))
        laplacian = np.zeros((n_points, n_points))
        for i in range(n_points):
            if self.cells[i].cell_type == CellType.DOMAIN:
                ones_idxs = i + np.tile(self.cell_strides, 2) * np.repeat(
                    np.array([-1, 1]), self.n_dim
                )
                valid_ones = ones_idxs[
                    np.logical_and(ones_idxs >= 0, ones_idxs < n_points)
                ]
                laplacian[i, valid_ones] = 1
                laplacian[i, i] = -4

            if self.cells[i].cell_type == CellType.GHOST:
                if self.is_corner(self.cells[i]):
                    laplacian[i, i] = -1
                else:
                    n_idx = self.ghost_neighbor(i)
                    laplacian[i, i] = -1
                    laplacian[i, n_idx] = 1

        return laplacian

    def is_corner(self, cell: Cell) -> bool:
        for i in range(self.n_dim):
            if not np.isclose(cell.center[i], self.domain[i][0]) or np.isclose(
                cell.center[i], self.domain[i][1]
            ):
                return False
        return True

    def ghost_neighbor(self, cell_idx: int) -> int:
        """
        Get the cell that is a neighbor to the given ghost cell
        """
        cell = self.cells[cell_idx]
        neighbor_idx = None
        for i in range(self.n_dim):
            if np.isclose(cell.center[i], self.domain[i][0] - self.dx[i] / 2):
                neighbor_idx = int(cell_idx + self.cell_strides[i])
            elif np.isclose(cell.center[i], self.domain[i][1] + self.dx[i] / 2):
                neighbor_idx = int(cell_idx - self.cell_strides[i])

        if neighbor_idx is None:
            raise ValueError("Not an edge cell")
        return neighbor_idx

    @functools.cached_property
    def cell_strides(self):
        """
        Strides for moving between cells in the grid
        :return:
        """
        return np.fromiter(
            (
                np.prod(self.n_cells) / np.prod(self.n_cells[i:])
                for i in range(self.n_dim)
            ),
            dtype=int,
        )

    @functools.cached_property
    def domain_cell_strides(self):
        """
        Strides for moving between cells in the domain, excluding ghost cells
        :return:
        """
        n_cells = tuple(n - (2 * int(self.use_ghost_nodes)) for n in self.n_cells)
        return np.fromiter(
            (np.prod(n_cells) / np.prod(n_cells[i:]) for i in range(self.n_dim)),
            dtype=int,
        )

    @functools.cached_property
    def ghost_node_mask(self):
        return np.array([cell.cell_type == CellType.GHOST for cell in self.cells])
