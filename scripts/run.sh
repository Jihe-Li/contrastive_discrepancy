gpu_id=4

for gaussian_num in 204800 102400 51200 25600 12800 6400 3200 1600 800 400 200;
do
CUDA_VISIBLE_DEVICES=$gpu_id, python run.py \
                                     network.max_densify_num=$gaussian_num
done
