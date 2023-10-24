import os
from tqdm import tqdm
import glob
import pandas
import numpy as np


class Voxceleb:
    def __init__(self, root: str = None, ext="mp4"):
        self.root = root
        self.length = len(glob.glob(f"{root}/**/**/**/*.{ext}"))
        self.table = None

    @staticmethod
    def __generator__(directory: str):
        all_dir = os.listdir(directory)
        for d in all_dir:
            yield d, directory + f'/{d}'

    def generator(self, number_id: int = None, number_part: int = None):
        for p, (part, part_root) in enumerate(self.__generator__(self.root)):
            if p == number_part:
                break
            for i, (id, id_root) in enumerate(self.__generator__(part_root)):
                if i == number_id:
                    break
                for ytb_id, ytb_id_root in self.__generator__(id_root):
                    for name, name_root in self.__generator__(ytb_id_root):
                        yield part, id, ytb_id, name, name_root

    def generate_table(self, number_id: int = None, number_part: int = None):
        part_list = []
        files_list = []
        ytb_id_list = []
        id_list = []
        name_list = []
        with tqdm(total=self.length, desc=f"Create table (VOXCELEB): ") as pbar:
            for part, id, ytb_id, name, name_root in self.generator(number_id=number_id, number_part=number_part):
                part_list.append(part)
                files_list.append(name_root)
                id_list.append(id)
                ytb_id_list.append(ytb_id)
                name_list.append(name)
                pbar.update(1)
        self.table = pandas.DataFrame(np.array([part_list, id_list,  ytb_id_list, name_list, files_list]).transpose(),
                                      columns=['part', 'id', 'ytb_id', 'name', 'file_path'])


if __name__ == '__main__':
    vox = Voxceleb(root=r"D:\These\data\Audio-Visual\voxceleb\train")
    vox.generate_table(number_id=None, number_part=None)
    print(vox.table['part'])

