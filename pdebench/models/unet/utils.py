

from __future__ import annotations

import math as mt
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class UNetDatasetSingle(Dataset):
    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        reduced_resolution=1,
        reduced_resolution_t=1,
        reduced_batch=1,
        if_test=False,
        test_ratio=0.1,
        num_samples_max=-1,
    ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY

        """

        # Define path to files
        root_path = Path(saved_folder + filename).resolve()
        assert filename[-2:] != "h5", "HDF5 data is assumed!!"

        with h5py.File(root_path, "r") as f:
            keys = list(f.keys())
            keys.sort()
            if "tensor" not in keys:
                _data = np.array(f["density"], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                if len(idx_cfd) == 3:  # 1D
                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            mt.ceil(idx_cfd[1] / reduced_resolution_t),
                            3,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    # pressure
                    _data = np.array(
                        f["pressure"], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    # Vx
                    _data = np.array(f["Vx"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch

                if len(idx_cfd) == 4:  # 2D
                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            idx_cfd[3] // reduced_resolution,
                            mt.ceil(idx_cfd[1] / reduced_resolution_t),
                            4,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    # pressure
                    _data = np.array(
                        f["pressure"], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    # Vx
                    _data = np.array(f["Vx"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch
                    # Vy
                    _data = np.array(f["Vy"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 1))
                    self.data[..., 3] = _data  # batch, x, t, ch

                if len(idx_cfd) == 5:  # 3D
                    self.data = np.zeros(
                        [
                            idx_cfd[0] // reduced_batch,
                            idx_cfd[2] // reduced_resolution,
                            idx_cfd[3] // reduced_resolution,
                            idx_cfd[4] // reduced_resolution,
                            mt.ceil(idx_cfd[1] / reduced_resolution_t),
                            5,
                        ],
                        dtype=np.float32,
                    )
                    # density
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 0] = _data  # batch, x, t, ch
                    # pressure
                    _data = np.array(
                        f["pressure"], dtype=np.float32
                    )  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 1] = _data  # batch, x, t, ch
                    # Vx
                    _data = np.array(f["Vx"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 2] = _data  # batch, x, t, ch
                    # Vy
                    _data = np.array(f["Vy"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 3] = _data  # batch, x, t, ch
                    # Vz
                    _data = np.array(f["Vz"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        ::reduced_resolution_t,
                        ::reduced_resolution,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data, (0, 2, 3, 4, 1))
                    self.data[..., 4] = _data  # batch, x, t, ch

            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f["tensor"], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[
                        ::reduced_batch, ::reduced_resolution_t, ::reduced_resolution
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                if len(_data.shape) == 4:  # 2D Darcy flow
                    # u: label
                    _data = _data[
                        ::reduced_batch, :, ::reduced_resolution, ::reduced_resolution
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    # if _data.shape[-1]==1:  # if nt==1
                    #    _data = np.tile(_data, (1, 1, 1, 2))
                    self.data = _data
                    # nu: input
                    _data = np.array(f["nu"], dtype=np.float32)  # batch, time, x,...
                    _data = _data[
                        ::reduced_batch,
                        None,
                        ::reduced_resolution,
                        ::reduced_resolution,
                    ]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                    self.data = np.concatenate([_data, self.data], axis=-1)
                    self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch

        if num_samples_max > 0:
            num_samples_max = min(num_samples_max, self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, ..., : self.initial_step, :], self.data[idx]


class UNetDatasetMult(Dataset):
    def __init__(
        self,
        filename,
        initial_step=10,
        saved_folder="../data/",
        if_test=False,
        test_ratio=0.1,
    ):
        """

        :param filename: filename that contains the dataset
        :type filename: STR
        :param filenum: array containing indices of filename included in the dataset
        :type filenum: ARRAY
        :param initial_step: time steps taken as initial condition, defaults to 10
        :type initial_step: INT, optional

        """

        # Define path to files
        self.file_path = Path(saved_folder + filename + ".h5").resolve()

        # Extract list of seeds
        with h5py.File(self.file_path, "r") as h5_file:
            data_list = sorted(h5_file.keys())

        test_idx = int(len(data_list) * (1 - test_ratio))
        if if_test:
            self.data_list = np.array(data_list[test_idx:])
        else:
            self.data_list = np.array(data_list[:test_idx])

        # Time steps used as initial conditions
        self.initial_step = initial_step

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Open file and read data
        with h5py.File(self.file_path, "r") as h5_file:
            seed_group = h5_file[self.data_list[idx]]

            # data dim = [t, x1, ..., xd, v]
            data = np.array(seed_group["data"], dtype="f")
            data = torch.tensor(data, dtype=torch.float)

            # convert to [x1, ..., xd, t, v]
            permute_idx = list(range(1, len(data.shape) - 1))
            permute_idx.extend([0, -1])
            data = data.permute(permute_idx)

        return data[..., : self.initial_step, :], data
