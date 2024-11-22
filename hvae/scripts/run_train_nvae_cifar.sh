
dataset=cifar10
num_latent_scales=1
num_groups_per_scale=30
num_latent_per_group=20
min_groups_per_scale=30


DATA_DIR=../../data
CHECKPOINT_DIR=output/$dataset/
EXPR_ID=ltnt_scl$num_latent_per_group-grp_per_scl$num_groups_per_scale-ltnt_per_grp$num_latent_per_group
# Note: args.save = CHECKPOINT_DIR/eval/save
# Models will be saved to args.save

python train_nvae.py --data $DATA_DIR/cifar10 --root $CHECKPOINT_DIR --save $EXPR_ID --dataset cifar10 \
        --num_channels_enc 128 --num_channels_dec 128 --epochs 150 --num_postprocess_cells 2 --num_preprocess_cells 2 \
        --num_latent_scales $num_latent_scales --num_latent_per_group $num_latent_per_group --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 \
        --num_preprocess_blocks 1 --num_postprocess_blocks 1 --num_groups_per_scale $num_groups_per_scale --batch_size 64 \
        --weight_decay_norm 1e-2 --num_nf 1 --num_process_per_node 1 --use_se --res_dist --fast --local_rank 4
