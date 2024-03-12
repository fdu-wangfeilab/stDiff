# stDiff: A Diffusion Model for Imputing Spatial Transcriptomics through Single-Cell Transcriptomics

A novel method named stDiff investigates the potential of employing diffusion models for single-cell omics generation.

## Framework
![framework](./framework.jpg)

## Arguments
### stDiff

if gene num < 512, batchsize = 2048, hiddensize = 512; \
if gene num > 512 and num < 1024, batchsize = 512, hiddensize = 1024; \
if gene num > 1024 and num < 2048, batchsize = 512, hiddensize = 2048. 

### baseline
The specific parameter settings follow those in the [Spatial Benchamark](https://github.com/QuKunLab/SpatialBenchmarking), and the baseline code is adapted from Spatial Benchmark, whose parameter settings also follow the default repositories of their respective models. \
The code for uniport is referenced in the example on its official website [Impute genes for MERFISH](https://uniport.readthedocs.io/en/latest/examples/MERFISH/MERFISH_impute.html). \
The code for SpatialScope is referenced in [SpatialScope for imputation](https://github.com/YangLabHKUST/SpatialScope/blob/master/demos/Mouse-MOp-MERFISH.ipynb). \


## How to run
### datasets
https://zenodo.org/records/10613236?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI5NDk0ZWE2LWMxZWEtNGU5ZS05ODQ1LTAyZTU5ZThhOTQ4NiIsImRhdGEiOnt9LCJyYW5kb20iOiJhODEzZjZhMDkxOTIwMjk5Y2I1ZjVkYzNmNWUwNjI0NyJ9.KX2t2EqEFatcx7PfIf_gkjmMkBjhknThlgJC3tuZOotwcDLwah8nqMvfy3uqmmiTariWNS3PLkZHjdIyzmqZPQ
### ckpt
Five sets of cross-validated checkpoints(random_state = 0) for all datasets have been uploaded to https://drive.google.com/drive/folders/1Qe1npltWkCuSZFBBG26cx7PUiM8K1Ghe?usp=sharing.
### environment
```bash
conda create -n stDiff python=3.6
conda activate stDiff
pip install -r requirements.txt
```
or
```bash
pip install stDiff-sc
```
### data preprocess
The datasets 2-16 in the experiment were all from the paper [Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type](https://www.nature.com/articles/s41592-022-01480-9). The raw data were initially processed in 'process/data_process.py' to convert the txt file to h5ad, and only the genes shared by both ST and scRNA-seq were retained. \

### run
test-stDiff for stDiff \
test-baseline for baselines \
test-spscope for SpatialScope

```python
from model.stDiff_model import DiT_stDiff
from model.stDiff_scheduler import NoiseScheduler
from model.stDiff_train import normal_train_stDiff
from model.sample import sample_stDiff
from process import *
```

```python
python test-stDiff.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'stDiff_result_name' --batch_size 512 --hidden_size 1024
python test-baseline.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'base_result_name'   
```
Use ```bash run.sh``` run both methods


### bug
Error is reported on RTX 3070, 3090 GPUs, the reason is that the default torch=1.9.0 cuda version does not match the GPU, after installing with pip, try 
```pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html```
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```