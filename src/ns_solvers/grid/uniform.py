"""
Creates a staggered uniform grid for FD navier stokes solvers
"""

import functools

import numpy as np
from plotly import graph_objects as go


class UniformGrid:
    def __init__(
        self,
        n_cells: tuple[int, ...],
        bounds: tuple[tuple[float, float], ...],
    ):
        self.n_cells = n_cells
        self.n_vertices = tuple(n + 1 for n in n_cells)
        self.bounds = bounds
        self.cell_dx = tuple((ub - lb) / n for (lb, ub), n in zip(bounds, n_cells))
        self.n_dim = len(self.n_vertices)

        self.strides_vertices = np.fromiter(
            (
                np.prod(self.n_vertices) / np.prod(self.n_vertices[i:])
                for i in range(self.n_dim)
            ),
            dtype=int,
        )

        self.strides_cells = np.fromiter(
            (
                np.prod(self.n_cells) / np.prod(self.n_cells[i:])
                for i in range(self.n_dim)
            ),
            dtype=int,
        )

        # N + 1 x ... vertices compared to N cells
        self.vertices: np.ndarray[float] = self._get_vertices()
        self.edge_vertices, self.edge_directions, self.edge_neighbors = (
            self._get_edge_vertices()
        )
        self.internal_edge_mask: np.ndarray[bool] = self._get_internal_edge_mask()

        self.neighbors = self._neighbors()

    @functools.cached_property
    def laplacian(self):
        n_points = int(np.prod(self.n_cells))
        laplacian = np.zeros((n_points, n_points))
        for i in range(n_points):
            ones_idxs = i + np.tile(self.strides_cells, self.n_dim) * np.repeat(
                np.array([-1, 1]), self.n_dim
            )
            valid_ones = ones_idxs[np.logical_and(ones_idxs >= 0, ones_idxs < n_points)]
            laplacian[i, valid_ones] = 1
            laplacian[i, i] = -4

        return laplacian

    @functools.cached_property
    def center_to_edge_neighbors(self) -> dict[str, np.ndarray[int]]:
        l_neighbors = np.concatenate(
            [
                np.arange(self.n_cells[0]) + i * self.n_vertices[0]
                for i in range(self.n_cells[0])
            ]
        )
        x_edges_count = int(np.sum(self.edge_directions == 0))
        r_neighbors = l_neighbors + self.strides_cells[0]
        b_neighbors = np.arange(np.prod(self.n_cells)) + x_edges_count
        t_neighbors = b_neighbors + self.strides_cells[1]
        return {  # type: ignore
            "left": l_neighbors,
            "right": r_neighbors,
            "bottom": b_neighbors,
            "top": t_neighbors,
        }

    def _get_internal_edge_mask(self) -> np.ndarray[bool]:
        """
        Get mask for internal edges
        :return:
        """
        edge_masks = []
        # for idx, edge_centers in enumerate(self.edge_vertices):
        for dim in range(self.n_dim):
            edge_centers = self.edge_vertices[:, self.edge_directions == dim]
            mask = np.zeros(edge_centers.shape[1], dtype=bool)
            for i in range(self.n_dim):
                mask = np.logical_or(
                    mask, np.isclose(edge_centers[i], self.bounds[i][0])
                )
                mask = np.logical_or(
                    mask, np.isclose(edge_centers[i], self.bounds[i][1])
                )
            edge_masks.append(~mask)
        return np.hstack(edge_masks)  # type: ignore

    def _get_vertices(self) -> np.ndarray[float]:
        """
        Generates vertices for a uniform grid
        :return:
        (x, y) coordinates of vertices
        """
        coord_gen = (np.linspace(*b, n + 1) for b, n in zip(self.bounds, self.n_cells))
        pos = np.vstack(list(map(np.ravel, np.meshgrid(*coord_gen))), dtype=float)
        return pos  # type: ignore

    def _get_edge_vertices(
        self,
    ) -> tuple[np.ndarray[float], np.ndarray[int], np.ndarray[int]]:
        """
        Generates vertices for a uniform grid
        :return:
        (x, y) coordinates of vertices
        """
        edge_vertices = []
        edge_directions = []
        edge_neighbors = []
        neighbor_offset = 0
        for ind in range(self.n_dim):
            coord_gen = (
                np.linspace(
                    lb + 0.5 * dx * int(idx != ind),
                    ub - 0.5 * dx * int(idx != ind),
                    n + int(idx == ind),
                )
                for idx, ((lb, ub), n, dx) in enumerate(
                    zip(self.bounds, self.n_cells, self.cell_dx)
                )
            )
            # Really new edge centers
            new_edges = np.vstack(
                list(map(np.ravel, np.meshgrid(*coord_gen))), dtype=float
            )
            edge_vertices.append(new_edges)
            edge_directions.append(np.ones(new_edges.shape[1], dtype=int) * ind)

            # Generate neighbor indices too
            # LR strides are 1, UD are n_cells
            # TODO idk if this will generalize to 3d
            strides = self.strides_cells[ind] * np.array([-1, 1])
            new_edge_neighbors = np.tile(np.arange(new_edges.shape[1]), 2).reshape(
                (2, -1)
            ) + strides.reshape((-1, 1))

            # mask out lower neighbors
            new_edge_neighbors[0] = np.where(
                np.isclose(new_edges[ind], self.bounds[ind][0]),
                -1,
                new_edge_neighbors[0],
            )
            # mask out upper neighbors
            new_edge_neighbors[1] = np.where(
                np.isclose(new_edges[ind], self.bounds[ind][1]),
                -1,
                new_edge_neighbors[1],
            )
            # apply offset for all edges being in one array
            new_edge_neighbors = np.where(
                new_edge_neighbors >= 0, new_edge_neighbors + neighbor_offset, -1
            )
            neighbor_offset += new_edges.shape[1]

            edge_neighbors.append(new_edge_neighbors)

        return (
            np.hstack(edge_vertices),
            np.hstack(edge_directions),
            np.hstack(edge_neighbors),
        )  # type: ignore

    def plot_vertices(self, fig: go.Figure | None = None, **plkw) -> go.Figure:
        fig = fig or go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.vertices[0].flatten(),
                y=self.vertices[1].flatten(),
                mode="markers",
                **plkw,
            )
        )
        return fig

    def plot_edge_vertices(
        self, fig: go.Figure | None = None, u: np.ndarray[float] | None = None, **plkw
    ) -> go.Figure:
        fig = fig or go.Figure()
        dim_names = ["x", "y", "z"]
        for dim in range(self.n_dim):
            edge_centers = self.edge_vertices[:, self.edge_directions == dim]
            text = None
            if u is not None:
                text = u[self.edge_directions == dim]
            fig.add_trace(
                go.Scatter(
                    x=edge_centers[0].flatten(),
                    y=edge_centers[1].flatten(),
                    mode="markers",
                    name=f"{dim_names[dim]}-edge centers",
                    text=text,
                    **plkw,
                )
            )
        return fig

    def plot_lines(self, fig: go.Figure, **plkw):
        assert len(self.n_vertices) == 2, "Plotting lines only supported in 2d"
        plot_fns = (fig.add_vline, fig.add_hline)
        for n_v, b, pfn in zip(self.n_vertices, self.bounds, plot_fns):
            for x in np.linspace(*b, n_v):
                pfn(x, line=dict(color="blue", width=0.5))
        return fig

    def _neighbors(self):
        n_vertices_total = self.vertices.shape[1]

        stride_array = np.repeat(self.strides_vertices, 2) * np.tile(
            np.array([-1, 1]), self.n_dim
        )
        neighbors = np.vstack([stride_array + i for i in range(n_vertices_total)])
        neighbors = np.where(
            np.logical_and(neighbors >= 0, neighbors < n_vertices_total), neighbors, -1
        )
        return neighbors


class UniformStaggeredGrid:
    """
    Create a grid of center points and a grid of
    """

    def __init__(
        self, n_cells: tuple[int, ...], bounds: tuple[tuple[float, float], ...]
    ):
        self.main_grid = UniformGrid(n_cells=n_cells, bounds=bounds)
        offset_consts = ((b[1] - b[0]) / (nc * 2) for b, nc in zip(bounds, n_cells))
        offset_bounds = tuple(
            (b[0] + oc, b[1] - oc) for b, oc in zip(bounds, offset_consts)
        )
        offset_n_cells = tuple(n - 1 for n in n_cells)
        self.offset_grid = UniformGrid(offset_n_cells, offset_bounds)

    def plot_vertices(self, u):
        fig = go.Figure()
        self.main_grid.plot_vertices(fig, name="main")
        self.main_grid.plot_edge_vertices(fig, u)
        self.main_grid.plot_lines(fig)
        self.offset_grid.plot_vertices(fig, name="interior")
        return fig
