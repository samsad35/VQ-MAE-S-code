import librosa


win_length = int(64e-3 * 16000)
# hop = int(0.625 * win_length)
hop = int(0.625 / 2 * win_length)


def griffin_lim(spec):
    signal = librosa.griffinlim(spec, hop_length=hop, win_length=win_length, n_iter=200)
    return signal
