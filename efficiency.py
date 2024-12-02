import os 
import torch
import numpy as np
import json
import time
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
from thop import profile
warnings.filterwarnings('ignore')
# args = option_trans.get_args_parser()
# args.resume_trans = '/home/adam/Desktop/T2M-RWKV/output/GPT/rwkv_net_best_fid.pth' #   /home/adam/Desktop/T2M-RWKV/pretrained/VQTransformer_corruption05/net_best_fid.pth
# args.block_size = 51
# trans_encoder = trans.Text2Motion_Transformer(
#                                 num_vq=args.nb_code, 
#                                 embed_dim=1024, 
#                                 clip_dim=args.clip_dim, 
#                                 block_size=args.block_size, 
#                                 num_layers=9, 
#                                 n_head=16, 
#                                 drop_out_rate=args.drop_out_rate, 
#                                 fc_rate=args.ff_rate
#                                 )
# print ('loading transformer checkpoint from {}'.format(args.resume_trans))
# ckpt = torch.load(args.resume_trans, map_location='cpu')
# trans_encoder.load_state_dict(ckpt['trans'], strict=True)
# trans_encoder.eval()
# trans_encoder.cuda()

trans_encoder = trans.Text2Motion_RNN(
                                PATH='/home/adam/Desktop/T2M-RWKV/output/rwkv_12/net_best_fid.pth',
                                RUN_DEVICE='cuda',
                                num_vq=512, 
                                embed_dim=768, 
                                clip_dim=512, 
                                block_size=51, 
                                num_layers=12, 
                                n_head=16, 
                                drop_out_rate=0.1, 
                                fc_rate=4)

bs = 1
dummy_input = torch.randn(bs, 512).cuda()

# warm
for i in range (50): 
    _ = trans_encoder.sample(dummy_input)

repetitions = 300
counts = np.zeros(repetitions)
# 创建 CUDA 事件对象
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for rep in range(repetitions):
    starter.record()
    _ = trans_encoder.sample(dummy_input, False)
    # trans_encoder.clear()
    ender.record()
    # 等待 GPU 同步
    torch.cuda.synchronize()
    curr_time = starter.elapsed_time(ender)
    counts[rep] = curr_time

# 计算平均推理时间和标准差
mean_time = np.mean(counts)
std_time = np.std(counts)
print(f"Mean Inference Time: {mean_time} ms")
print(f"Standard Deviation: {std_time} ms")
# # 计算吞吐量
# Throughput = (repetitions*batch_size)/np.sum(counts)
# print('Final Throughput:',Throughput)
# flops, params = profile(trans_encoder, (input,))
# print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))