import numpy as np
from PIL import Image
import os


def load_data(data_path):
    return np.load(data_path)

def trans_np_2_image(data_path, save_path):
    files = os.listdir(data_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for file in files:
        name = file.split('/')[-1]
        data = np.load(data_path+file)
        for i in range(len(data)):
            if np.any(data[i]):
                image = Image.fromarray(data[i])
                image = image.convert('L')
                image.save(save_path+ name.split('.')[0] + '.png')


if __name__ == '__main__':
    trans_np_2_image('../processed_data/train/0/', '../output/data_vis/train/')