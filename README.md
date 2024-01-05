# AAI-Final-Project

## Domain
This project is about invariant learning in DL. 

## Goal
We need to eliminate the influence of unstable feature and so on in model training and reasoning.

## Input & Output

- Input: modified MNIST images
  - add noise : assign wrong label to some images
  - add extra unstable/unrelated channels: $28 * 28$ -> $10 * 28 * 28$
    - The real greyscale channel may be one of the 10 channels, others channels are filled with 0s
    - For example, for some training data, the grayscale channel is the first channel, while the grayscale channel in other test data is the seventh channel (of course, it may also be other channel)
- Output: the correct label of the image

## Data Analysis for greyscale channel distribution
### Train data:
- row index: train label
- colomn index: channel index of greyscale data

```
   [[5365   54   73   61   67   73   66   46   64   60]
    [  59 5709   70   92   72   75   68   70   80   90]
    [  55   67 5435   80   77   62   61   61   73   72]
    [  66   76   76 5514   70   61   67   90   59   63]
    [  67   63   70   79 5275   65   73   84   59   55]
    [  80   44   55   56   48 4984   55   52   70   63]
    [  74   72   70   66   62   78 5354   60   59   66]
    [  65   67   80   57   63   77   72 5602   76   89]
    [  70   61   60   65   65   73   51   56 5347   63]
    [  74   72   74   60   71   77   55   55   62 5384]]
```

**The greyscale channel distribution of train data almost follows the labels.**


### Top 25 test data:

- check `0.npy`, `1.npy`, .... , `24.npy`
- greyscale data channels: `[7 6 9 0 7 8 3 4 6 8 0 7 8 4 1 8 7 4 3 5 8 1 6 7 5]`
- labels (mannual check) : `[6 0 5 4 9 9 2 1 9 4 8 7 3 9 7 4 4 4 9 2 5 4 7 6 7]`


**Obviously, the greyscale channel distribution of test data is different from train data.**


# Get Started

## Dependencies
> Recommanded: use Python virtual environment
>   ```bash
>       python -m venv venv
>       source venv/bin/activate # MacOS/Linux
>       # venv\Scripts\activate # Windows
>   ```

- For Windows with cuda
  ```bash
  pip install -r requirements_windows_cuda.txt
  ```
- For MacOS without cuda
  ```bash
  pip install -r requirements_macos.txt
  ```

## Options

You can see the options in `options.py`

## Train model

```bash
python train.py
```


- You can see the generated directory `checkpoints`, which save the model parameters, tensorboard logs and training options.
  - `checkpoints/{model_name}/[epoch]_[step].pth`: model parameters
  - `checkpoints/{model_name}/logs/`: tensorboard logs
  - `checkpoints/{model_name}/args.txt`: training options

- You can utilize the tensorboard logs to visualize the training process.
  ```bash
    tensorboard --logdir checkpoints/{model_name}/logs/
  ```





# Library / Package
- [Pytorch](https://pytorch.org/) is used for DL.
- [Tensorboard](https://www.tensorflow.org/tensorboard) is used for visualization.
- [torch_base](https://github.com/ahangchen/torch_base): DL Pytorch skeleton code
- [matplotlib](https://matplotlib.org/) is used for visualization.
- [tqdm](https://github.com/tqdm/tqdm) is used for progress bar.

# Reference

## MNIST ranking

- [paperwithcode](https://paperswithcode.com/sota/image-classification-on-mnist)
- [Reasonable Doubt: Get Onto the Top 35 MNIST Leaderboard by Quantifying Aleatoric Uncertainty](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)
 

## Papers

[1] Bao, Yujia, Shiyu Chang, and Regina Barzilay. "Learning stable classifiers by transferring unstable features." International Conference on Machine Learning. PMLR, 2022.

[2] Arjovsky, Martin, et al. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).

[3] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
