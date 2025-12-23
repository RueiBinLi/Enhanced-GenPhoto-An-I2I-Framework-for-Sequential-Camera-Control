# generative-photography

This project is derived from [[CVPR 2025 Highlight] Generative Photography](https://arxiv.org/abs/2412.02168).

> **Generative Photography: Scene-Consistent Camera Control for Realistic Text-to-Image Synthesis** <br>
> [Yu Yuan](https://yuyuan-space.github.io/), [Xijun Wang](https://www.linkedin.com/in/xijun-wang-747475208/), [Yichen Sheng](https://shengcn.github.io/), [Prateek Chennuri](https://www.linkedin.com/in/prateek-chennuri-3a25a8171/), [Xingguang Zhang](https://xg416.github.io/), [Stanley Chan](https://engineering.purdue.edu/ChanGroup/stanleychan.html)<br>

The original github page is https://github.com/pandayuanyu/generative-photography

## [[Paper](https://arxiv.org/abs/2412.02168)] [[Project Page](https://yuyuan-space.github.io/GenerativePhotography/)] [[Dataset](https://huggingface.co/datasets/pandaphd/camera_settings)] [[Weights](https://huggingface.co/pandaphd/generative_photography)] [[HF Demo](https://huggingface.co/spaces/pandaphd/generative_photography)]


## Configurations

### 1. Environment
*   CUDA 12.1, 64-bit Python 3.10 and PyTorch 2.1.1
*   Other environments may also work, at least for PyTorch 1.13.1 and CUDA 11.7
*   Users can use the following commands to install the packages
```bash
conda env create -f environment.yaml
conda activate genphoto
```

### 2. Prepare Models, Weights and Datasets
*   Install git-lfs first
```bash
sudo apt install git-lfs -y
```
*   Under I2I-generative-photography - Download models and weights
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/pandaphd/generative_photography
cd generative_photography
git lfs install
git lfs pull
cd ..
mv generative_photography models
```
*   Under I2I-generative-photography - Download dataset
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/pandaphd/camera_settings
cd camera_settings
git lfs install
git lfs pull
```

## Inference

### I2I Method Comparison

*   Using I2I method (**DDIM inversion** & **SDEdit**) to inference multiple camera settings
*   Input Image
  
![input image](input_image/my_park_photo.jpg)

*   Inference by DDIM inversion & SDEdit automatically
```bash
chmod +x run_ablation.sh
./run_ablation.sh
```

*   Result of DDIM Inversion

![DDIM result](images/stage1_bokeh_init_DDIM.gif)

*   Result of SDEdit

![SDEdit result](images/stage1_bokeh_init_SDEdit.gif)

### Multiple Stages

*   Use the validation set from the author to sample multiple camera settings
*   We divide it to 4 stages. In each stage, we add a new camera setting to the image

```bash
chmod +x run_experiment_runner.sh
./run_experiment_runner.sh
```

## Evaluation

*   We use Correlation Coefficient, LPIPS and CLIP to evaluate.

### I2I Method Comparison

```bash
python evaluation_code/evaluate_DDIM_SDEdit.py \
  --root_dir "Experiments/ablation_3" \
  --base_image "input_image/my_park_photo.jpg" \
  --prompt "A photo of a park with green grass and trees" \
  --output_dir "Evaluation_Results_test"
```

* Result

### Multiple Stages

```bash
python evaluation_code/evaluate_all.py \
  --root_dir "experiments_final" \
  --data_root "." \
  --output_dir "Evaluation_Results_test"
```

* Result
