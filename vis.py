import torch
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import options.option_transformer as option_trans
args = option_trans.get_args_parser()
args.dataname = 't2m'
args.resume_pth = 'pretrained/VQVAE/net_last.pth'
args.resume_trans = '/home/adam/Desktop/T2M-RWKV/output/GPT/rwkv_net_best_fid.pth' #  '' /home/adam/T2M-GPT/output/GPT/rwkv_net_last.pth
args.down_t = 2
args.depth = 3
args.block_size = 51

# change the text here
clip_text = [
            #  "A man walks forward, stumbles to the right, and then regains his balance and keeps walking forwards.",
            #  "A person has their forearms raised in front of them, then lowers them.",
            #  "A person is running.",
            #  "A person is walking while laughing.",
            #  "A person is doing jumping jacks."
            "The person does a salsa dance."
            #  "A person grabbed the leg and did something",
            #  "A man steps forward and does a handstand.",
            #  "A man rises from the ground, walks in a circle and sits back down on the ground.",
            #  "This person kicks with their right leg then jabs several times.",
            #  "A person walks in a clockwise circle and stops where he began.",
            #  "A person walks forward then turns completely around and does a cartwheel."
             ]
gif_path = './RWKV'
if not os.path.exists(gif_path):
    os.makedirs(gif_path)

# load models
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

import models.vqvae as vqvae
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)
print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

import models.t2m_trans as trans
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
                                PATH=args.resume_trans,
                                RUN_DEVICE='cuda',
                                num_vq=args.nb_code, 
                                embed_dim=1024, 
                                clip_dim=args.clip_dim, 
                                block_size= args.block_size, 
                                num_layers=18, 
                                n_head=16, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate
                                )
for p in trans_encoder.parameters():
    p.requires_grad = False

# for visualization
mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

from utils.motion_process import recover_from_ric

text = clip.tokenize(clip_text, truncate=True).cuda()    
texts_feature = clip_model.encode_text(text).float()
# for text_i in text:
#     texts_feature.append(clip_model.encode_text(text_i.unsqueeze(0)).float())
xyz_list = []
pred_idx = []
for input in texts_feature:
    input = input.unsqueeze(0) #(512) -> (B=1, 512)
    index_motion = trans_encoder.sample(input, if_categorial=True)
    pred_idx.append(index_motion.cpu().numpy())
    # trans_encoder.clear() # For RNN-Mode, clear states cache after each sequence
    pred_pose = net.forward_decoder(index_motion)
    pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22) 
    xyz = pred_xyz.reshape(-1, 22, 3)
    xyz = xyz.detach().cpu().numpy()
    xyz_list.append(xyz)

# with open('/home/adam/Desktop/T2M-RWKV/rnn_bat_preds.txt', 'w') as file:
#     for i in range(len(pred_idx)):
#         file.write(f'{pred_idx[i]}\n')
# np.save(f'{path}/{clip_text[0]}.npy', xyz.detach().cpu().numpy())

import visualization.plot_3d_global as plot_3d
pose_vis = plot_3d.draw_to_batch(xyz_list, clip_text, [f'{gif_path}/{text}.gif' for text in clip_text])