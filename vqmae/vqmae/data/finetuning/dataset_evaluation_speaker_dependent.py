from .ravdess import Ravdess
from .savee import Savee
from .emodb import EmoDB
from .mead import Mead
from .emovo import Emovo
from .iemocap import Iemocap
import torch
import numpy as np
from pathlib import Path
import librosa
from torch.utils.data import Dataset
import h5py
from sklearn.utils import shuffle
import pandas


class EvaluationDatasetSpeakerDependent(Dataset):
    def __init__(self,
                 root: str,
                 ratio_train: int = 80,
                 frames_per_clip: int = 50,
                 train: bool = True,
                 dataset: str = "ravdess",
                 h5_path: str = None
                 ):
        super().__init__()
        if dataset.lower() == "ravdess":
            self.data = Ravdess(root=Path(root))
        elif dataset.lower() == "emodb":
            self.data = EmoDB(root=Path(root))
        elif dataset.lower() == "savee":
            self.data = Savee(root=Path(root))
        elif dataset.lower() == "mead":
            self.data = Mead(root=Path(root))
        elif dataset.lower() == "emovo":
            self.data = Emovo(root=Path(root))
        elif dataset.lower() == "iemocap":
            self.data = Iemocap(root=Path(root))
        self.data.generate_table()
        self.table = self.data.table
        table = shuffle(self.table)
        if train:
            table.to_pickle("table.pkl")
            self.table = table[:int(len(self.table) * ratio_train / 100)].reset_index(drop=True)
        else:
            table = pandas.read_pickle("table.pkl")
            self.table = table[int(len(self.table) * ratio_train / 100):].reset_index(drop=True)

        print(f"\t --> Evaluation of {dataset.upper()} | train: {train} | length: {len(self.table)}")
        # -----
        self.h5_path = h5_path
        self.h5_bool = h5_path is not None
        self.number_frames = 0
        self.current_frame = 0
        self.seq_length = frames_per_clip
        win_length = int(64e-3 * 16000)
        hop = int(0.625 / 2 * win_length)
        self.spec_parameters = dict(n_fft=1024,
                                    hop=hop,
                                    win_length=win_length)

    def __len__(self):
        return len(self.table)

    def save_table(self, path: str):
        self.table.to_pickle(path)

    def to_spec(self, wav):
        X = librosa.stft(wav,
                         n_fft=self.spec_parameters["n_fft"],
                         hop_length=self.spec_parameters["hop"],
                         win_length=self.spec_parameters["win_length"])  # pad_mode = 'reflected'
        magnitude = np.abs(X)
        phase = np.angle(X)
        return magnitude, phase

    @staticmethod
    def load_wav(file: str):
        wav, sr = librosa.load(path=file, sr=16000)
        wav = librosa.to_mono(wav.transpose())
        wav = wav / np.max(np.abs(wav))
        return wav

    def get_information(self, index):
        path = self.table.iloc[index]['path']
        id = self.table.iloc[index]['id']
        emotion = self.table.iloc[index]['emotion']
        name = self.table.iloc[index]['name']
        return emotion, id, path, name

    @staticmethod
    def padding(data, seq_length=50):
        """

        :param seq_length:
        :param data:
        :return:
        """
        if len(data.shape) == 2:
            data = np.pad(data, ((0, seq_length - data.shape[0]), (0, 0)), 'wrap')
        return data

    def open(self):
        self.hdf5 = h5py.File(self.h5_path, mode='r')

    def read(self, id, name):
        m = np.array(self.hdf5[f'/{id}/{name}'])
        return m

    def __getitem__(self, item):
        if not hasattr(self, 'hdf5'):
            self.open()
        emotion, id, path, name = self.get_information(item)
        if self.h5_bool:
            spec = self.read(id, name)
        else:
            audio = self.load_wav(str(path))
            spec, _ = self.to_spec(audio)
            spec = (spec ** 2).transpose()
        number_frames = spec.shape[0]
        if number_frames < self.seq_length:
            spec = self.padding(spec, seq_length=self.seq_length)
        spec = torch.from_numpy(spec)
        if number_frames > self.seq_length:
            self.i_1 = spec[:self.seq_length]
        else:
            self.i_1 = spec[:self.seq_length]
        return self.i_1.type(torch.LongTensor), emotion
