# stDiff: A Diffusion Model for Imputing Spatial Transcriptomics through Single-Cell Transcriptomics

A novel method named stDiff investigates the potential of employing diffusion models for single-cell omics generation.

## Framework
![framework](./framework.jpg)

## Arguments
if gene num > 512 and num < 1024, batchsize = 512, hiddensize = 1024;
if gene num < 512, batchsize = 2048, hiddensize = 512;

## How to run
### environment
```bash
conda env create -f environment.yml
conda activate stDiff
pip install -r requirements.txt
```
### run
test-stDiff for stDiff \
test-baseline for baselines 

```python
python test-stDiff.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'stDiff_result_name' --batch_size 512 --hidden_size 1024
python test-baseline.py --sc_data 'sc_dataset(h5ad)' --sp_data 'sp_dataset(h5ad)' --document 'base_result_name'   
```
Use ```bash run.sh``` run both methods



