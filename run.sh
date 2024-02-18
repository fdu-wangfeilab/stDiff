# python test-stDiff.py --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_stDiff' --batch_size 512 --hidden_size 1024
# python test-stDiff.py --sc_data 'dataset3_seq_42.h5ad' --sp_data 'dataset3_spatial_42.h5ad' --document 'dataset3_stDiff_save_d100' --batch_size 2048 --hidden_size 512 --step 100
# python test-stDiff.py --sc_data 'dataset3_seq_42.h5ad' --sp_data 'dataset3_spatial_42.h5ad' --document 'dataset3_stDiff_save_d1000' --batch_size 2048 --hidden_size 512 --step 1000
# python test-stDiff.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_stDiff_save_d4500' --batch_size 2048 --hidden_size 512 --step 4500
# python test-stDiff.py --sc_data 'dataset3_seq_42.h5ad' --sp_data 'dataset3_spatial_42.h5ad' --document 'dataset3_stDiff_save_d10000' --batch_size 2048 --hidden_size 512 --step 10000
# python test-stDiff.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_stDiff_save_e100' --batch_size 2048 --hidden_size 512 --epoch 100
# python test-stDiff.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_stDiff_save_e300' --batch_size 2048 --hidden_size 512 --epoch 300
# python test-stDiff.py --sc_data 'dataset3_seq_42.h5ad' --sp_data 'dataset3_spatial_42.h5ad' --document 'dataset3_stDiff_save_1000' --batch_size 2048 --hidden_size 512 --step 1000
# python test-stDiff.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_stDiff_save_e3000' --batch_size 2048 --hidden_size 512 --epoch 3000
# python test-stDiff.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_stDiff_save_e1500' --batch_size 2048 --hidden_size 512 --epoch 1500
# python test-stDiff.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_stDiff_save_e9000' --batch_size 2048 --hidden_size 512 --epoch 9000



#!/bin/bash

# 生成4个随机整数
random_numbers=()
for i in {1..4}; do
    random_numbers+=($((RANDOM % 100 + 1)))
done

# 打印生成的随机数
# echo "生成的随机数: ${random_numbers[0]}"
echo ${random_numbers[0]} ${random_numbers[1]} ${random_numbers[2]} ${random_numbers[3]}


# python test-stDiff.py  --sc_data 'dataset3_seq_42.h5ad' --sp_data 'dataset3_spatial_42.h5ad' --document 'dataset3_stDiff_save' --batch_size 2048 --hidden_size 512
# python test-stDiff.py  --sc_data 'dataset4_seq_1000.h5ad' --sp_data 'dataset4_spatial_1000.h5ad' --document 'dataset4_stDiff_rand1' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[0]}
# python test-stDiff.py  --sc_data 'dataset4_seq_1000.h5ad' --sp_data 'dataset4_spatial_1000.h5ad' --document 'dataset4_stDiff_rand2' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[1]}
# python test-stDiff.py  --sc_data 'dataset4_seq_1000.h5ad' --sp_data 'dataset4_spatial_1000.h5ad' --document 'dataset4_stDiff_rand3' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[2]}
# python test-stDiff.py  --sc_data 'dataset4_seq_1000.h5ad' --sp_data 'dataset4_spatial_1000.h5ad' --document 'dataset4_stDiff_rand4' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[3]}
# 66 2 74 13
python test-stDiff.py  --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_stDiff_rand1' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[0]}
python test-stDiff.py  --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_stDiff_rand2' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[1]}
python test-stDiff.py  --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_stDiff_rand3' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[2]}
python test-stDiff.py  --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_stDiff_rand4' --batch_size 512 --hidden_size 1024 --rand ${random_numbers[3]}

# python test-stDiff.py  --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_stDiff_save' --batch_size 512 --hidden_size 1024 
# python test-stDiff.py  --sc_data 'dataset6_seq_251.h5ad' --sp_data 'dataset6_spatial_251.h5ad' --document 'dataset6_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset7_seq_118.h5ad' --sp_data 'dataset7_spatial_118.h5ad' --document 'dataset7_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset8_seq_84.h5ad' --sp_data 'dataset8_spatial_84.h5ad' --document 'dataset8_stDiff_save' --batch_size 2048 --hidden_size 512
# python test-stDiff.py  --sc_data 'dataset9_seq_76.h5ad' --sp_data 'dataset9_spatial_76.h5ad' --document 'dataset9_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset10_seq_42.h5ad' --sp_data 'dataset10_spatial_42.h5ad' --document 'dataset10_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset11_seq_347.h5ad' --sp_data 'dataset11_spatial_347.h5ad' --document 'dataset11_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset12_seq_1000.h5ad' --sp_data 'dataset12_spatial_1000.h5ad' --document 'dataset12_stDiff_save' --batch_size 512 --hidden_size 1024
# python test-stDiff.py  --sc_data 'dataset13_seq_154.h5ad' --sp_data 'dataset13_spatial_154.h5ad' --document 'dataset13_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset14_seq_981.h5ad' --sp_data 'dataset14_spatial_981.h5ad' --document 'dataset14_stDiff_save' --batch_size 512 --hidden_size 1024 
# python test-stDiff.py  --sc_data 'dataset15_seq_141.h5ad' --sp_data 'dataset15_spatial_141.h5ad' --document 'dataset15_stDiff_save' --batch_size 2048 --hidden_size 512 
# python test-stDiff.py  --sc_data 'dataset16_seq_118.h5ad' --sp_data 'dataset16_spatial_118.h5ad' --document 'dataset16_stDiff_save' --batch_size 2048 --hidden_size 512 

# python test-baseline.py --sc_data 'dataset5_seq_915.h5ad' --sp_data 'dataset5_spatial_915.h5ad' --document 'dataset5_base'  
# python test-baseline.py --sc_data 'dataset2_seq_33.h5ad' --sp_data 'dataset2_spatial_33.h5ad' --document 'dataset2_base'  

