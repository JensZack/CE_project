import os
from shutil import rmtree
import vtk
from vtk.util import numpy_support as VN  # noqa
from pathlib import Path
import re
from subprocess import run
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import Dataset
from copy import copy
from torch import Tensor
import torch
import functools


class VTKParser:

    def __init__(self, data_dir: Path, prefix: str = 'solution'):
        self.data_dir = data_dir
        self.dx, self.dims, self.points = self.vtk_metadata(data_dir / 'solution_t=0p0.vtr')
        self._new_dims = tuple(int(d/2) for d in self.dims[:2])
        self.files = sorted(self.data_dir.glob(f'{prefix}*.vtr'), key=lambda x: self.get_timestamp(x) or -1.0)

    @staticmethod
    def get_timestamp(file: Path, time_re: str = r"t=(\d+p\d+)"):
        match = re.search(time_re, file.name)
        if match:
            return float(match.group(1).replace("p", "."))
        return None

    @staticmethod
    def vtk_metadata(filename: Path):
        assert filename.resolve().exists(), f'File not found: {filename}'
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(str(filename))
        reader.Update()

        data = reader.GetOutput()
        dim = data.GetDimensions()
        points = np.array([data.GetPoint(i) for i in range(data.GetNumberOfPoints())])
        dx = tuple(1/d for d in dim)
        return dx, dim, points

    def h5_to_numpy(self, filename: Path | None = None):
        filename = filename or self.data_dir / 'ns_velocity.h5'
        hf = h5py.File(filename, 'r')
        return hf['velocity'][:], hf['time'][:]


    def vtk_to_numpy(self, filename: Path) -> np.ndarray:
        assert filename.resolve().exists(), f'File not found: {filename}'
        reader = vtk.vtkXMLRectilinearGridReader()
        reader.SetFileName(str(filename))
        reader.Update()

        data = reader.GetOutput()
        u = VN.vtk_to_numpy(data.GetPointData().GetArray('velocity'))
        top_quad = slice(None, int(self._new_dims[0]))
        return u.reshape(*self.dims[:2], 2)[top_quad, top_quad, :]


    def vtk_collection_to_numpy(self):

        for file in self.files:
            ts = self.get_timestamp(file)
            if ts:
                yield ts, self.vtk_to_numpy(file)

    def numpy_to_hdf5(self, fileout: Path):
        hf = h5py.File(fileout, 'w')
        n_frames = len(self.files) - 1
        vel_arr = np.empty((*self._new_dims, 2, n_frames), dtype=float)
        idx_arr = np.empty(n_frames, dtype=int)
        time_arr = np.empty(n_frames, dtype=float)
        for idx, (ts, vel) in enumerate(self.vtk_collection_to_numpy()):
            vel_arr[..., idx] = vel
            idx_arr[idx] = idx
            time_arr[idx] = ts
        if np.any(np.isnan(vel_arr)):
            raise ValueError("Zero velocity field found, check for errors in vtk_to_numpy")
        hf.create_dataset('velocity', data=vel_arr)
        hf.create_dataset('t_idx', data=idx_arr)
        hf.create_dataset('time', data=time_arr)
        hf.close()

    def velocity_to_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dvdx = np.gradient(v, axis=1) / self.dx[0]
        dudy = np.gradient(u, axis=0) / self.dx[1]
        return dvdx - dudy

    def vorticity_plot_ts(self):
        ts_gen = self.vtk_collection_to_numpy()
        fig, ax = plt.subplots()

        ts, velocity = next(ts_gen)
        u, v = velocity[..., 0], velocity[..., 1]
        vorticity = self.velocity_to_vorticity(u, v)

        img = ax.imshow(vorticity, label='vorticity')
        ax.set_title(f'Time: {ts} s')

        def update(frame):
            # for each frame, update the data stored on each artist.
            try:
                ts_, velocity_ = next(ts_gen)
            except StopIteration:
                return (img,)

            ax.set_title(f'Time: {ts_} s')
            u_, v_ = velocity_[..., 0], velocity_[..., 1]
            vorticity_ = self.velocity_to_vorticity(u_, v_)
            img.set_data(vorticity_)

            return (img,)

        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, repeat=True)
        plt.show()

    def plot_points(self):
        fig, ax = plt.subplots()
        ax.scatter(self.points[:, 0], self.points[:, 1], c=np.arange(len(self.points), dtype=float))
        plt.show()


def run_simulation():
    target_dir = Path("/Users/zackjensen/repos/IncompressibleNavierStokes.jl/examples")
    target = target_dir / "DecayingTurbulence2D.jl"
    cmd = ["julia", "+1.9.4", f"--project={target_dir}", str(target)]
    run(cmd)


def move_sim(target: Path = Path("/Users/zackjensen/data/DecayingTurbulence2D")):
    sim_count = len(os.listdir(target))
    mv_target = target / f"sim_{sim_count + 1}"
    mv_target.mkdir()
    sim_output = Path("/Users/zackjensen/repos/IncompressibleNavierStokes.jl/examples/output/DecayingTurbulance2D")
    sim_output.rename(mv_target)
    return mv_target


def run_simulations(n: int):
    i = 0
    while i < n:
        print(f"Running simulation {i+1}")
        run_simulation()
        sim_dir = move_sim()
        parser = VTKParser(sim_dir)
        try:
            parser.numpy_to_hdf5(sim_dir / 'ns_velocity.h5')
        except ValueError:
            rmtree(sim_dir)
            continue
        i += 1


class SimDataset(Dataset):

    def __init__(self,
                 root_dir: Path,
                 frames_per_h5: int = 100,
                 sim_ixds: list[int] | None = None,
                 resolution: int = 128,
             ):
        self.resolution = resolution
        self.sim_idxs = sim_ixds
        self.frames_per_h5 = frames_per_h5
        self.root_dir = root_dir
        self.files = sorted(root_dir.glob('sim_*/ns_velocity.h5'))
        self._frames = self._load_h5()

    def set_sim_idxs(self, sim_ixds: list[int]):
        self.sim_idxs = sim_ixds

    @property
    def frames(self):
        if self.sim_idxs is not None:
            return [self._frames[i] for i in self.sim_idxs]
        return self._frames

    @frames.setter
    def frames(self, value):
        self._frames = value

    def _load_h5(self):
        frames = []
        for idx, file in enumerate(self.files):
            with h5py.File(file, 'r') as hf:
                vel = hf['velocity'][:]
                frames.append(torch.from_numpy(vel).swapaxes(0, 3).swapaxes(1, 2).to(torch.float32))

        return frames

    def __len__(self):
        return len(self.frames) * (self.frames_per_h5 - 1)

    def subsample_mask(self, offset: tuple[int, int] | None = None):
        # base resolution is 256, 256
        assert self.resolution in [256, 128, 64, 32, 16], "Resolution must be one of [128, 64, 32, 16]"
        scalar = 256 // self.resolution
        idxs = np.arange(0, 256, scalar)

        if offset is None:
            # for now, use an offset that puts the stensel roughly in the middle of each frame
            offset = (scalar // 2, scalar // 2)

        idx_x = idxs + offset[0]
        mask_x = np.zeros(256, dtype=bool)
        mask_x[idx_x] = True
        idx_y = idxs + offset[1]
        mask_y = np.zeros(256, dtype=bool)
        mask_y[idx_y] = True

        mask = np.ones((256, 256), dtype=bool)
        mask[~mask_x, :] = False
        mask[:, ~mask_y] = False

        return mask


    def __getitem__(self, item):
        h5_idx = item // (self.frames_per_h5 - 1)
        frame_idx = item % (self.frames_per_h5 - 1)

        mask = self.subsample_mask()

        return {
            "x": self.frames[h5_idx][frame_idx][:, mask].reshape((2, self.resolution, self.resolution)),
            "y": self.frames[h5_idx][frame_idx + 1][:, mask].reshape((2, self.resolution, self.resolution))
        }

    @classmethod
    def train_test_datasets(
            cls,
            root_dir: Path,
            frames_per_h5: int = 100,
            train_ratio: float = 0.8):
        dataset = cls(root_dir, frames_per_h5)
        train_idx = np.random.choice(len(dataset.frames), int(len(dataset.frames) * train_ratio), replace=False)
        test_idx = np.setdiff1d(np.arange(len(dataset.frames)), train_idx)

        dataset.set_sim_idxs(list(train_idx))
        test_dataset = copy(dataset)
        test_dataset.set_sim_idxs(list(test_idx))

        return dataset, test_dataset

    def vorticity_plot_ts(self, vf: Tensor):
        fig, ax = plt.subplots()

        idx = 0
        dt = .01
        vorticity = velocity_to_vorticity(vf[idx])

        img = ax.imshow(vorticity, label='vorticity')
        ax.set_title(f'Time: {(idx * dt):.4f} s')

        def update(frame):
            nonlocal idx
            idx += 1
            # for each frame, update the data stored on each artist.
            if idx >= vf.shape[0]:
                return (img,)

            ax.set_title(f'Time: {(idx * dt):.4f} s')
            vorticity_ = velocity_to_vorticity(vf[idx])
            img.set_data(vorticity_)

            return (img,)

        ani = animation.FuncAnimation(fig=fig, func=update, interval=50, repeat=True)
        plt.show()


def velocity_to_vorticity(vf: Tensor) -> np.ndarray:
    dx = 1 / (vf.shape[1] - 1)

    dvdx = torch.gradient(vf[0, ...], axis=1)[0] / dx
    dudy = torch.gradient(vf[1, ...], axis=0)[0] / dx
    return (dvdx - dudy).numpy()

