# stDiff: A Diffusion Model for Imputing Spatial Transcriptomics through Single-Cell Transcriptomics

A novel method named stDiff investigates the potential of employing diffusion models for single-cell omics generation.

## Framework
![framework](./framework.jpg)

## Arguments
### stDiff
if gene num > 512 and num < 1024, batchsize = 512, hiddensize = 1024;
if gene num < 512, batchsize = 2048, hiddensize = 512;
### baseline
The specific parameter settings follow those in the [Spatial Benchamark](https://github.com/QuKunLab/SpatialBenchmarking), and the baseline code is adapted from Spatial Benchmark, whose parameter settings also follow the default repositories of their respective models.
The code for uniport is referenced in the example on its official website [Impute genes for MERFISH](https://uniport.readthedocs.io/en/latest/examples/MERFISH/MERFISH_impute.html).


## How to run
### ckpt
Five sets of cross-validated checkpoints for all datasets have been uploaded to https://drive.google.com/file/d/1oOSBm1cP0J5jYgiH3HNrgs1EDJS53YRR/view?usp=drive_link.
### environment
```bash
conda env create -f environment.yml
conda activate stDiff
pip install -r requirements.txt
```
### data preprocess
The datasets 2-16 in the experiment were all from the paper [Benchmarking spatial and single-cell transcriptomics integration methods for transcript distribution prediction and cell type](https://www.nature.com/articles/s41592-022-01480-9). The raw data were initially processed in 'process/data_process.py' to convert the txt file to h5ad, and only the genes shared by both ST and scRN-seq were retained.

### run
test-stDiff for stDiff \
test-baseline for baselines 

```python
python test-stDiff.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'stDiff_result_name' --batch_size 512 --hidden_size 1024
python test-baseline.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'base_result_name'   
```
Use ```bash run.sh``` run both methods



