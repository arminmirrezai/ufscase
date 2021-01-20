import matplotlib.pyplot as plt
from pathlib import Path
import os


class VisualizeData:

    def __init__(self, df):
        self.df = df

    def figure(self, keyword, save=False):
        self.df[self.df.keyword == keyword].plot(x='startDate', y='interest', label=keyword)
        plt.ylabel('Relative search volume')
        plt.xlabel('Weeks')
        if save:
            folder_path = Path.cwd().absolute().parents[0].as_posix() + "/Data/Graphs/" + self.df.country.unique()[0]
            self._createDir(folder_path)
            file_name = keyword + ".png"
            plt.savefig(folder_path + "/" + file_name)
        plt.show()

    @staticmethod
    def _createDir(path):
        """
        Create directory of multiple folders
        """
        folders = []
        curr_path = path
        while not os.path.exists(curr_path):
            if curr_path == '':
                break
            curr_path, folder = os.path.split(curr_path)
            folders.append(folder)
        for i in range(len(folders) - 1, -1, -1):
            curr_path += '/' + folders[i]
            try:
                os.mkdir(curr_path)
            except OSError:
                print("Failed to create folder: " + folders[i])









