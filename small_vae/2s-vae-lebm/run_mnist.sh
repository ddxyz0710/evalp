dt=0000
vaedir="../vae-baseline/output/train_mnist/RAE-Baseline/0000"
netGpth=$vaedir/netG_bestfid.pth
netIpth=$vaedir/netI_bestfid.pth
python train_cifar.py --datetime $dt --gpu 0 --workers 4 \
    --data_to_0_1 True --netG $netGpth --netI $netIpth  