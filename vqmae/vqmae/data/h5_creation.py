import h5py
from ..model import SpeechVQVAE
from .finetuning import EvaluationDataset
from tqdm import tqdm
import torch
import librosa
import numpy as np
from ..tools import load_audio_from_video, to_spec

torch.cuda.empty_cache()

win_length = int(64e-3 * 16000)
spec_parameters = dict(n_fft=1024,
                       hop=int(0.625 / 2 * win_length),
                       win_length=win_length)


def load_wav(file: str):
    wav, sr = librosa.load(path=file, sr=16000)
    wav = librosa.to_mono(wav)
    wav = wav / np.max(np.abs(wav))
    return wav


def h5_creation(vqvae: SpeechVQVAE,
                dataset: EvaluationDataset,
                dir_save: str
                ):
    data_ = dataset.data
    file_h5 = h5py.File(dir_save, 'a')
    vqvae.to('cuda')
    with tqdm(total=len(dataset.table)) as pbar:
        for id, name, path, emotion in data_.generator():
            pbar.update(1)
            pbar.set_description(f"ID: {id}, name: {name}")
            # Get indices for each file .mp4
            if file_h5.get(f'/{id}/{name}'):
                continue
            data = load_wav(file=str(path))
            data, _ = to_spec(data, spec_parameters)
            data = torch.from_numpy((data ** 2).transpose()).type(torch.FloatTensor).unsqueeze(1)
            try:
                data = data.to("cuda")
                indices = vqvae.get_codebook_indices(data).cpu().detach().numpy()
            except RuntimeError:
                pass
            # Create the path in H5 file:
            if not file_h5.get(f'/{id}'):
                file_h5.create_group(f'/{id}')
            group_temp = file_h5[f'/{id}']
            # Save indices in H5 file
            image_h5 = group_temp.create_dataset(name=name, data=indices, dtype="int32")
            image_h5.attrs.create('id', id)

    # Close h5 file
    file_h5.flush()
    file_h5.close()
