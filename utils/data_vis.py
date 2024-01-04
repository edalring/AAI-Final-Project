import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

from utils.torch_utils import load_npy_as_ndarray



# 展示图片
def export_to_greyscale_png(image_array, save_path, use_plt=False):
    if use_plt:
        # plt绘制的图更好看，不过更慢
        plt.imshow(image_array, cmap='gray')  # 如果是灰度图，使用 cmap='gray'
        plt.imshow(image_array, cmap='viridis')  # 使用颜色显示灰度图
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(save_path)
    else:

        datum_scaled = (image_array * 255).astype(np.uint8)

        # Convert to image
        image = Image.fromarray(datum_scaled)
        image.save(save_path)

def save_datum_as_image(path_pair):
    datum_path, output_path = path_pair
    datum = load_npy_as_ndarray(datum_path)
    # datum is 10 * 28 * 28, sum up to 28 * 28
    datum = datum.sum(axis=0)

    export_to_greyscale_png(datum, output_path)


def trans_np_2_image(data_path, save_path):
    data_root = Path(data_path)
    save_root = Path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)

    data_path = list(data_root.rglob('*.npy'))

    def get_output_path(input_path: Path):
        rel_path = input_path.relative_to(data_root)
        out_dir = save_root / rel_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / (rel_path.name.rstrip('.npy') + '.png')


    output_paths = map(get_output_path, data_path)

    path_pairs = list(zip(data_path, output_paths))

    with Pool() as pool:
        list(tqdm(pool.imap(save_datum_as_image, path_pairs), total=len(path_pairs)))


    
if __name__ == '__main__':
    trans_np_2_image('../processed_data/', '../output/data_vis/')