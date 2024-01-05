import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

'''
## Data Analysis for greyscale channel distribution
### Train data:
- row index: train label
- colomn index: channel index of greyscale data

    [[5365.   54.   73.   61.   67.   73.   66.   46.   64.   60.]
    [  59. 5709.   70.   92.   72.   75.   68.   70.   80.   90.]                                                                                                                                                                                                              
    [  55.   67. 5435.   80.   77.   62.   61.   61.   73.   72.]
    [  66.   76.   76. 5514.   70.   61.   67.   90.   59.   63.]
    [  67.   63.   70.   79. 5275.   65.   73.   84.   59.   55.]
    [  80.   44.   55.   56.   48. 4984.   55.   52.   70.   63.]
    [  74.   72.   70.   66.   62.   78. 5354.   60.   59.   66.]
    [  65.   67.   80.   57.   63.   77.   72. 5602.   76.   89.]
    [  70.   61.   60.   65.   65.   73.   51.   56. 5347.   63.]
    [  74.   72.   74.   60.   71.   77.   55.   55.   62. 5384.]]

**The greyscale channel distribution of train data almost follows the labels.**


### Top 25 test data:

- check `0.npy`, `1.npy`, .... , `24.npy`
- greyscale data channels: [7 6 9 0 7 8 3 4 6 8 0 7 8 4 1 8 7 4 3 5 8 1 6 7 5]
- labels (mannual check) : [6 0 5 4 9 9 2 1 9 4 8 7 3 9 7 4 4 4 9 2 5 4 7 6 7]


**Obviously, the greyscale channel distribution of test data is different from train data.**
'''


def multi_processes_execute(task, data, workers=None, use_tqdm=True):
    with Pool(processes=workers) as pool:
        data_ret = pool.imap(task, data)
        if use_tqdm:
            data_ret = tqdm(data_ret, total=len(data), position=1, leave=False)
        return list(data_ret)


def resolve_greyscale_channel(file):
    data = np.load(file)
    
    # data is 10 * 28 * 28, get the channel index where not all data is 0
    channel_index = np.where(np.any(data, axis=(1, 2)))[0]
    return channel_index[0]


def check_data(data_path):
    data_path = Path(data_path)
    count = np.zeros((10, 10))
    for c in tqdm(range(10), position=0, leave=False):
        data_sub_path = data_path / str(c)
        files = list(data_sub_path.rglob('*.npy'))

        idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)
        idx = np.array(idx)

        # bincount
        for i in range(10):
            count[c, i] = np.sum(idx == i)

        

    print(count)
    print(np.sum(count, axis=0))
    print(np.sum(count, axis=1))

def check_test_data(data_path, num=25):
    data_path = Path(data_path) / 'test'

    files = list(data_path.iterdir())
    files.sort(key= lambda x: int(x.name.rstrip('.npy')))

    files = files[:num]

    idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)

    idx = np.array(idx)

    print(files)
    print(idx)    

if __name__ == '__main__':
    # check_data('../processed_data/train')
    check_test_data('../processed_data/')


