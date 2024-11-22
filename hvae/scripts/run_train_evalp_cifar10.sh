
dataset=cifar10
CHECKPOINT_DIR=output/$dataset/eval-ltnt_scl20-grp_per_scl30-ltnt_per_grp20
DATA_DIR=$CHECKPOINT_DIR/feats
EXPR_ID=eval-ltnt_scl20-grp_per_scl30-ltnt_per_grp20
num_groups_per_scale_list="30" 
mid_chnls=40
num_blks=4
lrD=3e-4
lrF=5e-5
klwt=1e-8
wtdcay=0.
seed=42
disc_dir=output/$dataset/evalp/$EXPR_ID/ebmConv4_1-Fnet2_smallInCouplng-nscl2--beta1.5-midchn$mid_chnls-nblk$num_blks-lrD$lrD-lrF$lrF-klwt$klwt-wtdcay$wtdcay/seed$seed


 python train_evalp_cifar10.py --data_dir $DATA_DIR/ \
    --root_dir ./ \
    --checkpoint $CHECKPOINT_DIR/checkpoint.pt \
    --save $disc_dir --is_local --eval_on_train \
    --num_groups_per_scale_list $num_groups_per_scale_list \
    --group_idx 0 --train_batch_size 50 --num_fid_samples 20000 \
    --sampler_mid_channels $mid_chnls --sampler_num_blocks $num_blks \
    --lrD $lrD --lrF $lrF --kl_weight $klwt --weight_decay $wtdcay\
    --seed $seed \
