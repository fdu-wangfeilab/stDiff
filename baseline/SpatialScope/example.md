## preprocess
baseline/SpatialScope/data_preprocess.ipynb process scRNA-seq data

## train dataset
python ./baseline/SpatialScope/Train_scRef.py --ckpt_path ./ckpt/dataset9 --scRef ./ckpt/sc/dataset9_seq_76_spscope.h5ad --cell_class_column subclass_label --gpus 0 --bs 512 --factors 3,4,5,1 --epoch 5000

## run to impute
python test-spscope.py --sc_data 'dataset9_seq_76.h5ad' --sp_data 'dataset9_spatial_76.h5ad' --document 'dataset9_spscope'