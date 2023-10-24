#!/usr/bin/env python3
import librosa
import numpy as np
import matplotlib.pyplot as plt
import warnings
from math import *
from scipy.io.wavfile import write
import sounddevice as sd
from numba import jit, cuda
import librosa.display as display
warnings.filterwarnings(action='ignore')


class AudioTools:
    def __init__(self, audio_config: dict = None):
        # == ==
        self.audio_config = audio_config
        # == ==
        self.sampling_rate = self.audio_config['sampling_rate']
        self.win_length = int(self.audio_config['win_length'] * self.audio_config['sampling_rate'])
        # self.win_length = np.int(np.power(2, np.ceil(np.log2(self.win_length))))
        self.hop = int(self.audio_config['hop_percent'] * self.win_length)
        # self.hop = self.audio_config['hop_percent'] * self.win_length
        self.n_fft = self.win_length
        self.win = np.sin(np.arange(.5, self.win_length - .5 + 1) / self.win_length * np.pi)

    def load_audio(self, file_name: str = "", play: bool = False) -> tuple:
        signal, sampling_rate = librosa.load(str(file_name), sr=self.sampling_rate)
        signal = signal/np.max(np.abs(signal))
        signal, index = librosa.effects.trim(signal, top_db=30)
        if play:
            sd.play(signal, self.sampling_rate)
        return signal, sampling_rate

    # @jit(target="cuda")
    def stft(self, wave_audio) -> tuple:
        X = librosa.stft(wave_audio, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length)  # pad_mode = 'reflected'
        magnitude = np.abs(X)
        phase = np.angle(X)
        self.X = X
        return magnitude, phase

    def istft(self, stft_matrix):
        signal = librosa.istft(stft_matrix, hop_length=self.hop, win_length=self.win_length)
        # signal = librosa.istft(stft_matrix)
        return signal

    def mel(self, wave_audio):
        mel = librosa.feature.melspectrogram(wave_audio, n_fft=self.nfft, win_length=self.wlen, hop_length=self.hop,
                                             window=self.win)
        return mel

    def griffin_lim(self, S, **kwargs):
        signal = librosa.griffinlim(S, hop_length=self.hop, win_length=self.win_length, **kwargs)
        return signal

    def write(self, signal, name: str = "temp.wav"):
        write(name, self.sampling_rate, signal)

    def plot_spectrogram(self, spec, show=True, save: str = None):
        plt.style.use('default')
        fig, ax = plt.subplots()
        # img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time')
        img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=self.sampling_rate, y_axis='linear',
                               x_axis='time', hop_length=self.hop)
        plt.xlabel('Time (s)', fontsize=15)
        plt.ylabel('Frequency (Hz)', fontsize=15)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        if save is not None:
            plt.savefig(save)

        if show:
            plt.show(block=True)
        else:
            plt.close()

        return fig

    def play_audio(self, signal):
        sd.play(signal, self.sampling_rate)

    def pitch_tracking(self, y: np.ndarray, method=librosa.pyin):
        f0, voiced_flag, voiced_prob = \
            librosa.pyin(y=y, fmin=65, fmax=2093, sr=self.sampling_rate, win_length=self.win_length, hop_length=self.hop)
        return f0, voiced_flag, voiced_prob


# ======================================================================================================================
if __name__ == '__main__':
    audio = AudioTools(audio_config={'win_length': 64e-3, 'hop_percent': 0.25, 'sampling_rate': 16000})
    signal1, rate = audio.load_audio(file_name=r'D:\These\data\Audio-Visual\MEAD\Train\M03\audio\angry\level_1\001.m4a',
                                     play=True)
    magnitude, phase = audio.stft(signal1)
    audio.plot_spectrogram(magnitude)
    print(magnitude.shape)

    f0, voiced_flag, voiced_prob = audio.pitch_tracking(signal1)
    print(f0.shape)
    plt.plot(voiced_flag)
    plt.show()

    signal2 = audio.istft(audio.X)
    audio.write(signal2, name='istft.wav')
    #
    signal3 = audio.griffin_lim(magnitude, n_iter=1000)
    audio.write(signal3, name='griffin.wav')


