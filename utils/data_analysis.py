import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import shutil
import json


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

### Valid data:
- row index: valid label
- colomn index: index of valid data file
- NaN: the valid data file does not exist

 [ 0.  0.  6.  2.  3.  3.  6. nan nan nan nan nan nan nan nan nan]                                                   
 [ 5.  7.  5.  0.  6.  1.  1.  4.  8.  3. nan nan nan nan nan nan]
 [ 6.  5.  1.  6.  0.  5.  0.  1. nan nan nan nan nan nan nan nan]
 [ 0.  6.  4.  2.  6.  3.  6. nan nan nan nan nan nan nan nan nan]
 [ 9.  0.  3.  7.  2.  0.  7.  1.  7.  3.  6.  4. nan nan nan nan]
 [ 3.  7.  2.  9.  3.  5.  0.  5.  3.  3.  7.  8. nan nan nan nan]
 [ 1.  9.  8.  3.  9.  5.  9.  7.  3.  3. nan nan nan nan nan nan]
 [ 6.  3.  1.  0.  0.  3.  1.  3.  1.  2.  9.  8.  9.  7.  7.  2.]
 [ 7.  5.  9. nan nan nan nan nan nan nan nan nan nan nan nan nan]
 [ 7.  5.  9.  9.  8.  4.  4.  9.  1.  5.  7.  2.  5.  2.  3. nan]]


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


def check_train_data(data_path):
    data_path = Path(data_path) / 'train'
    count = np.zeros((10, 10))

    label_2_check = 0
    channel_2_check = 2
    CHECK_SAVE_ROOT = data_path.parent.parent / 'output' / 'check_vis' / f'label{label_2_check}_channel{channel_2_check}'

    CHECK_SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    PNG_ROOT  = data_path.parent.parent / 'output' / 'data_vis' / 'train' / f'{label_2_check}'

    sample_channels = None
    sample_files = None
    for c in tqdm(range(10), position=0, leave=False):
        data_sub_path = data_path / str(c)
        files = list(data_sub_path.rglob('*.npy'))
        files.sort(key= lambda x: int(x.name.rstrip('.npy')) )


        idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)
        idx = np.array(idx)

        if c == label_2_check:
            sample_channels = idx
            sample_files = files

        # bincount
        for i in range(10):
            count[c, i] = np.sum(idx == i)

    for file, idx in zip(sample_files, sample_channels):
        if idx == channel_2_check:
            png_file = PNG_ROOT / file.name.replace('npy', 'png')
            shutil.copy(png_file, CHECK_SAVE_ROOT / png_file.name)

    print(count)
    print(np.sum(count, axis=0))
    print(np.sum(count, axis=1))

def check_val_data(data_path):
    data_path = Path(data_path) / 'val'

    files = list(data_path.rglob('*.npy'))


    files.sort(key= lambda x: int(x.name.rstrip('.npy')) + int(x.parent.name)  * 1000 )


    idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)


    data = [[] for _ in range(10)]
    for channel, file in zip(idx, files):
        data[int(file.parent.name)].append(channel)

    # align the length
    max_len = max([len(d) for d in data])
    for d in data:
        d.extend([np.nan] * (max_len - len(d)))
    
    data = np.array(data)
    print(data)

def check_test_data(data_path, num=25):
    data_path = Path(data_path) / 'test'

    files = list(data_path.iterdir())
    files.sort(key= lambda x: int(x.name.rstrip('.npy')))

    idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)


    count = np.zeros(10)
    for i in range(10):
        count[i] = np.sum(np.array(idx) == i)
    
    print(count)


    files = files[:num]

    idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)

    idx = np.array(idx)

    print(files)
    print(idx)    

def divide_train_dataset(train_data_path):
    idx_tabel = {}
    for i in range(10):
        data_sub_path = Path(train_data_path) / str(i)
        files = list(data_sub_path.rglob('*.npy'))
        files.sort(key= lambda x: int(x.name.rstrip('.npy')) )
        idx = multi_processes_execute(resolve_greyscale_channel, files, use_tqdm=True)
        idx_tabel.update({str(i):{}})
        for k in range (10):
            idx_tabel[str(i)].update({str(k):[]})
        for channel, file in zip(idx, files):
            idx_tabel[str(i)][str(channel)].append(file.parent.name+'\\'+file.name)

    with open('./data.json', 'w') as json_file:
        json.dump(idx_tabel, json_file)



if __name__ == '__main__':
    # check_train_data('../processed_data/')
    # check_test_data('../processed_data/')
    # check_val_data('../processed_data/')
    divide_train_dataset('../processed_data/train/')


