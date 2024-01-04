# AAI-Final-Project

## Domain
This project is about invariant learning in DL. 

## Goal
We need to eliminate the influence of unstable feature and so on in model training and reasoning.

## Input & Output

- Input: modified MNIST images
  - add noise : assign wrong label to some images
  - add extra unstable/unrelated channels: $28 * 28$ -> $10 * 28 * 28$
- Output: the correct label of the image
  
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
- tqdm: is used for progress bar.

# Reference

## MNIST ranking

- [paperwithcode](https://paperswithcode.com/sota/image-classification-on-mnist)
- [Reasonable Doubt: Get Onto the Top 35 MNIST Leaderboard by Quantifying Aleatoric Uncertainty](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)
 

## Papers

[1] Bao, Yujia, Shiyu Chang, and Regina Barzilay. "Learning stable classifiers by transferring unstable features." International Conference on Machine Learning. PMLR, 2022.

[2] Arjovsky, Martin, et al. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).

[3] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
