# AAI-Final-Project

## Domain
This project is about invariant learning in DL. 

## Goal
We need to eliminate the influence of unstable feature and so on in model training and reasoning.

## Input & Output

- Input: modified MNIST images
  - add noise : assign wrong label to some images
  - add extra unstable/unrelated channels: $28 \times 28$ -> $10 \times 28 \times 28$
    - The real greyscale channel may be one of the 10 channels, others channels are filled with 0s
    - For example, for some training data, the grayscale channel is the first channel, while the grayscale channel in other test data is the seventh channel (of course, it may also be other channel)
- Output: the correct label of the image

## Data Analysis for greyscale channel distribution
### Train Data:
- row index: train label
- colomn index: channel index of greyscale data
- data[i,j] : the number of train data with label i and greyscale channel j

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

### Valid Data:
- row index: valid label
- colomn index: index of valid data file
- "-": the valid data file does not exist
- data[i,j] : the greyscale channel index of valid data with label i and index j

```
[[ 0  0  6  2  3  3  6  -  -  -  -  -  -  -  -  -]                                                
 [ 5  7  5  0  6  1  1  4  8  3  -  -  -  -  -  -]
 [ 6  5  1  6  0  5  0  1  -  -  -  -  -  -  -  -]
 [ 0  6  4  2  6  3  6  -  -  -  -  -  -  -  -  -]
 [ 9  0  3  7  2  0  7  1  7  3  6  4  -  -  -  -]
 [ 3  7  2  9  3  5  0  5  3  3  7  8  -  -  -  -]
 [ 1  9  8  3  9  5  9  7  3  3  -  -  -  -  -  -]
 [ 6  3  1  0  0  3  1  3  1  2  9  8  9  7  7  2]
 [ 7  5  9  -  -  -  -  -  -  -  -  -  -  -  -  -]
 [ 7  5  9  9  8  4  4  9  1  5  7  2  5  2  3  -]]
```

**Obviously, the greyscale channel distribution of valid data is different from train data.**


### Test Data:
- Greyscale channel distribution of test data is uniform.
  - `greyscale_channel_count[i]` : the number of test data with greyscale channel `i`
  ```Python
  greyscale_channel_count = [935 1037  980  989 1028 1018  946  983 1013  971]
  ```

- Top 25 test data
  - check `0.npy`, `1.npy`, .... , `24.npy`
  - greyscale data channels: `[7 6 9 0 7 8 3 4 6 8 0 7 8 4 1 8 7 4 3 5 8 1 6 7 5]`
  - labels (mannual check) : `[6 0 5 4 9 9 2 1 9 4 8 7 3 9 7 4 4 4 9 2 5 4 7 6 7]`


**Obviously, the greyscale channel distribution of test data is different from train data.**


# Get Started
## Prepare

- Put your data in <repo>/processed_data, if you obey the default configure
  - Otherwise, specify your data directory by `--data_path=...` (see [options.py](https://github.com/edalring/AAI-Final-Project/blob/main/options.py))

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

You can see the options in [`options.py`](https://github.com/edalring/AAI-Final-Project/blob/main/options.py)

## Train Model

- Train model with our methodology for Invariant Feature Learning
    ```bash
    python train.py --model_type=[xxx] # vgg by default
    ```
- Train model directly (regardless of unstable features)
    ```bash
    python train.py --straight_forward --model_type=[xxx] # vgg by default
    ```

### Test Model

- Test model on the test set to output prediction file

  ```bash
  python test.py --load_model_path=[saved model path]
  ```

## Data Access


- You can see the generated directory `checkpoints`, which save the model parameters, tensorboard logs and training options.
  - `checkpoints/{model_name}/[epoch]_[step].pth`: model parameters
  - `checkpoints/{model_name}/logs/`: tensorboard logs
  - `checkpoints/{model_name}/args.txt`: training options

- You can utilize the tensorboard logs to visualize the training process.
  ```bash
    tensorboard --logdir checkpoints/{model_name}/logs/
  ```

- While training, the train loss / train acc / valid loss / valid acc would be printed out in the console per epoch. You can also use tensorboard to export these data via web UI.





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
 

## Repository

- [Tofu](https://github.com/YujiaBao/Tofu): The code for this work "Learning stable classifiers by transferring unstable features."
- [ExquisiteNetV2](https://github.com/shyhyawJou/ExquisiteNetV2/tree/main): one of the SOTA small model for MNIST (99.71% acc with 518230 params, ref:  [paperwithcode](https://paperswithcode.com/sota/image-classification-on-mnist))



## Papers

[1] Bao, Yujia, Shiyu Chang, and Regina Barzilay. "Learning stable classifiers by transferring unstable features." International Conference on Machine Learning. PMLR, 2022.

[2] Arjovsky, Martin, et al. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).

[3] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
