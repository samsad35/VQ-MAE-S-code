from moviepy.editor import *


def load_audio_from_video(path_video: str = None,
                          path_audio_save: str = None,
                          sr: int = None):
    video = VideoFileClip(path_video, audio_fps=sr)
    audio = video.audio
    # print(video.reader.nframes)
    if path_audio_save is not None:
        audio.write_audiofile(path_audio_save)
    return audio.to_soundarray()


# if __name__ == '__main__':
#     audio = load_audio_from_video(
#         path_video=r"D:\These\data\Audio-Visual\voxceleb\train\parta\id00177\6jwTxn5f2Wc\00074.mp4",
#         path_audio_save=None)
#     # s, fs = librosa.load(path="test.wav")
#     # print(fs)
#     print(audio.to_soundarray())
