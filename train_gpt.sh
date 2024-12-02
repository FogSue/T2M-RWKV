# 6e-4 for L12-D768, 4e-4 for L24-D1024
# CUDA_VISIBLE_DEVICES=7 
python3 train_t2m_trans.py  \
--exp-name v7_0.5uncond \
--batch-size 128 \
--num-layers 12 \
--embed-dim-gpt 768 \
--nb-code 512 \
--n-head-gpt 16 \
--block-size 51 \
--ff-rate 4 \
--drop-out-rate 0.05 \
--resume-pth pretrained/VQVAE/net_last.pth \
--vq-name VQVAE \
--out-dir output \
--total-iter 300000 \
--lr-scheduler 150000 \
--lr 0.00001 \
--dataname t2m \
--down-t 2 \
--depth 3 \
--quantizer ema_reset \
--eval-iter 1000 \
--pkeep 0.5 \
--dilation-growth-rate 3 \
--vq-act relu