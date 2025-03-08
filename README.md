# Energy-based Variational Latent Prior (EVaLP)

This repository contains the code for implementating the experiments in paper titled "Learning Energy-Based Variational Latent Priors for VAEs"  

## Setup  

Install the required packages  
  ``pip install -r requirements.txt``  

## Train VAEs with single latent variables  

### 1st Stage  

Train the VAE model  

- Change directory: ``cd small_vae/vae-baseline``
- Run the CelebA experiment: ``bash run_celeba.sh``
- Run the CIFAR experiment: ``bash run_cifar.sh``
- Run the MNIST experiment: ``bash run_mnist.sh``

### 2nd Stage  

Train the EVaLP model:  

- Change directory: ``cd small_vae/2s-vae-lebm``
- Run the CelebA64 experiment: ``bash run_celeba.sh``
- Run the CIFAR10 experiment: ``bash run_cifar.sh``
- Run the MNIST experiment: ``bash run_mnist.sh``

## Train VAEs with hierarchical latent vairables  

- Change directory to vae-baseline: ``cd hvae``
- Precompute FID statistics: ``python precompute_fid_statistics.py --data data/celeba64_lmdb --dataset celeba_64 --fid_dir fid-stats/``

### 1st Stage  

Train the HVAE model:  

- Run the CelebA experiment: ``bash scripts/run_train_nvae_celeba64.sh``
- Run the CIFAR experiment: ``bash scripts/run_train_nvae_cifar.sh``  

### 2nd Stage  

- Run the CelebA64 experiment: ``bash scripts/run_train_evalp_celeba64.sh``  
- Run the CIFAR10 experiment: ``bash scripts/run_train_evalp_cifar.sh``
