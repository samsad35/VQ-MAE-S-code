import matplotlib.pyplot as plt
import librosa
import librosa.display as display
import numpy as np
win_length = int(64e-3 * 16000)
# hop = int(0.625 * win_length)
hop = int(0.625 / 2 * win_length)


def plot_spectrogram(spec, show=True, save: str = None):
    plt.style.use('default')
    fig, ax = plt.subplots()
    # img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), y_axis='log', x_axis='time')
    img = display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=16000, y_axis='linear',
                           x_axis='time', hop_length=hop, win_length=win_length)
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
