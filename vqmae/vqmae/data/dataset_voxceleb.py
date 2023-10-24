from abc import ABC
from .base import BASE
from .voxceleb import Voxceleb
import torch
import random
import numpy as np
import h5py
import pandas


class VoxcelebSequential(BASE, ABC):
    def __init__(self,
                 root: str,
                 frames_per_clip: int = 50,
                 hop_length: float = 100,
                 train: bool = True,
                 table_path: str = None,
                 h5_path: str = None):
        super().__init__()
        if table_path is None:
            self.vox = Voxceleb(root=root)
            self.vox.generate_table(number_id=None)
            self.table = self.vox.table
        else:
            self.table = pandas.read_pickle(table_path)
        self.h5_path = h5_path
        self.h5_bool = True
        # -----
        self.train = train
        self.list_ = list(np.arange(0, len(self.table)))
        self.number_frames = 0
        self.current_frame = 0
        self.shuffle = True
        self.seq_length = frames_per_clip
        self.hop_length = hop_length
        random.shuffle(self.list_)

    def __len__(self):
        return len(self.table) // 10  # 126896  # 1712

    def save_table(self, path: str):
        self.table.to_pickle(path)

    def open(self):
        self.hdf5 = h5py.File(self.h5_path, mode='r')

    def read(self, indx: tuple):
        m = np.array(self.hdf5[f'/{indx[0]}/{indx[1]}/{indx[2]}/{indx[3]}'])
        return m

    def get_information(self, index):
        part = self.table.iloc[index]['part']
        id = self.table.iloc[index]['id']
        ytb_id = self.table.iloc[index]['ytb_id']
        name = self.table.iloc[index]['name']
        return part, id, ytb_id, name

    def __getitem__(self, item):
        if not hasattr(self, 'hdf5'):
            self.open()
        while True:
            info = self.get_information(self.list_[item])
            try:
                self.modality = self.read(info)
            except:
                item += 1
                continue
            self.modality = torch.from_numpy(self.modality)
            self.number_frames = self.modality.shape[0]
            if self.number_frames >= self.seq_length:
                break
            else:
                item += 1
        self.current_frame = np.random.randint(0, self.number_frames - self.seq_length)
        self.i_1 = self.modality[self.current_frame: self.current_frame + self.seq_length]
        # return self.i_1.type(torch.LongTensor)
        return self.i_1.type(torch.FloatTensor)
