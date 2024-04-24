srun --mpi=pmi2 -p basemodel  -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=8 --quotatype=auto \
    python -u /mnt/lustre/zhengjinliang/See-what-you-need-to-see/model/llava_encoder.py