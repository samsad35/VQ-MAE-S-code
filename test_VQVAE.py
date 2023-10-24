import torch
from vqmae import SpeechVQVAE, VoxcelebSequential
import hydra
from omegaconf import DictConfig
import os
import librosa
import numpy as np
from scipy.io.wavfile import write


win_length = int(64e-3 * 16000)
hop = int(0.625 / 2 * win_length)
spec_parameters = dict(n_fft=1024,
                       hop=int(0.625 / 2 * win_length),
                       win_length=win_length)


def griffin_lim(spec):
    signal = librosa.griffinlim(spec, hop_length=hop, win_length=win_length, n_iter=500)
    return signal


def to_spec(wav, spec_parameters: dict):
    X = librosa.stft(wav,
                     n_fft=spec_parameters["n_fft"],
                     hop_length=spec_parameters["hop"],
                     win_length=spec_parameters["win_length"])  # pad_mode = 'reflected'
    magnitude = np.abs(X)
    phase = np.angle(X)
    return magnitude, phase


def load_wav(file: str):
    wav, sr = librosa.load(path=file, sr=16000)
    wav = librosa.to_mono(wav.transpose())
    wav = wav / np.max(np.abs(wav))
    return wav


def istft(stft_matrix, phase):
    convstft = stft_matrix * (np.cos(phase) + 1j * np.sin(phase))
    signal = librosa.istft(convstft, hop_length=hop, win_length=win_length)
    return signal


def save_wav(audio, save: str = None, vqvae: SpeechVQVAE = None, phase=None):
    # audio = vqvae.decode(indices[0])
    audio = np.sqrt(torch.transpose(audio.squeeze(1), 0, 1).cpu().detach().numpy())
    if phase is not None:
        signal = istft(audio, phase)
    else:
        signal = griffin_lim(audio)
    write(save, 16000, signal)


@hydra.main(config_path="config_mae", config_name="config")
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    """ Data """
    wav = load_wav(file=r"D:\These\data\Audio\emovo\f1\dis-f1-b2.wav")
    spec, phase = to_spec(wav, spec_parameters)
    spec = (spec ** 2).transpose()

    """ VQVAE """
    vqvae = SpeechVQVAE(**cfg.vqvae)
    vqvae.load(path_model=r"checkpoint/SPEECH_VQVAE/2022-12-27/21-42/model_checkpoint")
    _, x_recon, _ = vqvae(torch.from_numpy(spec).unsqueeze(1))
    save_wav(x_recon, save="out.wav", phase=phase)
    wav = griffin_lim(np.sqrt(spec).transpose())
    write(r"real.wav", 16000, wav)


if __name__ == '__main__':
    main()
