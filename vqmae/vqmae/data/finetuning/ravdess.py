import os
from tqdm import tqdm
import glob
import pandas
import numpy as np
from pathlib import Path


class Ravdess:
    def __init__(self, root: Path = None, ext="wav"):
        self.root = root
        self.length = len(glob.glob(f"{root}/**/*.{ext}"))
        self.table = None

    @staticmethod
    def __generator__(directory: Path):
        all_dir = os.listdir(directory)
        for d in all_dir:
            yield d, directory / d

    def generator(self):
        for id, id_root in self.__generator__(self.root):
            for name, path in self.__generator__(id_root):
                emotion = int(name.split(".")[0].split("-")[2])-1
                yield id, name, path, emotion

    def generate_table(self):
        files_list = []
        id_list = []
        name_list = []
        emotion_list = []
        with tqdm(total=self.length, desc=f"Create table (RAVDESS): ") as pbar:
            for id, name, path, emotion in self.generator():
                files_list.append(path)
                id_list.append(id)
                emotion_list.append(emotion)
                name_list.append(name)
                pbar.update(1)
        self.table = pandas.DataFrame(np.array([id_list, name_list,  files_list, emotion_list]).transpose(),
                                      columns=['id', 'name', 'path', 'emotion'])


if __name__ == '__main__':
    vox = Ravdess(root=Path(r"D:\These\data\Audio\RAVDESS"))
    vox.generate_table()
    print(vox.table["id"])

