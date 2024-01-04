import numpy as np
from pathlib import Path
from tqdm import tqdm

def check_data(data_path):
    data_path = Path(data_path)
    for c in tqdm(range(10)):
        data_sub_path = data_path / str(c)
        print(data_sub_path)
        files = list(data_sub_path.rglob('*.npy'))
        count = np.zeros(10)

        for file in files:
            data = np.load(file)
            for i in range(10):
                if np.any(data[i]):
                    count[i] += 1
                    break
        

        print("The channel distribution for class{}, is {}".format(c, count))

if __name__ == '__main__':
    check_data('../processed_data/train')


