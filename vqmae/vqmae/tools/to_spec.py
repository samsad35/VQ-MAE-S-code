import librosa
import numpy as np


def to_spec(wav, spec_parameters: dict):
    X = librosa.stft(wav,
                     n_fft=spec_parameters["n_fft"],
                     hop_length=spec_parameters["hop"],
                     win_length=spec_parameters["win_length"])  # pad_mode = 'reflected'
    magnitude = np.abs(X)
    phase = np.angle(X)
    return magnitude, phase
