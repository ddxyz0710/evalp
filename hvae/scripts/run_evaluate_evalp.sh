CHECKPOINT_DIR=output/cifar10/eval-ltnt_scl20-grp_per_scl30-ltnt_per_grp20
discdir=<dir of discriminator checkpoint>
DATA_DIR=../../data
save_dir=$discdir/eval
echkpt=$discdir/Dnet.pth
fchkpt=$discdir/Fnet.pth
nblks=2
f_mid_chn_dim=40
multidim=100

python evaluate_evalp_nvae.py --eval_mode='sample' \
    --checkpoint $CHECKPOINT_DIR/checkpoint.pt \
    --f_checkpoint $fchkpt --e_checkpoint $echkpt \
    --sampler_mid_channels $f_mid_chn_dim --sampler_num_blocks $nblks \
    --data $DATA_DIR/celeba64_lmdb \
    --fid_dir fid-stats/ --temp=0.5 --readjust_bn \
    --save $save_dir --sir_multinomial_dim $multidim \
    --batch_size 100 \
