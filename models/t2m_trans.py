import math, os
import torch
import copy
import types
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import logging
logger = logging.getLogger(__name__)
import random
import models.pos_encoding as pos_encoding

os.environ['RWKV_FLOAT_MODE'] = 'bf16'
T_MAX = 51
from torch.utils.cpp_extension import load
wkv_cuda = load(name="wkv", sources=["./models/cuda/wkv_op.cpp", "./models/cuda/wkv_cuda.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}'])

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
        else:
            w = -torch.exp(w.float().contiguous())
            u = u.float().contiguous()
            k = k.float().contiguous()
            v = v.float().contiguous()
        ctx.save_for_backward(w, u, k, v)
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return y
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return y.half()
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return y.bfloat16()

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.contiguous(), gw, gu, gk, gv)
        else:
            wkv_cuda.backward(B, T, C, w, u, k, v, gy.float().contiguous(), gw, gu, gk, gv)
        gw = torch.sum(gw, dim=0)
        gu = torch.sum(gu, dim=0)
        if '32' in os.environ['RWKV_FLOAT_MODE']:
            return (None, None, None, gw, gu, gk, gv)
        elif os.environ['RWKV_FLOAT_MODE'] == 'fp16':
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif os.environ['RWKV_FLOAT_MODE'] == 'bf16':
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())

def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())

'''
RWKV-v4 with RNN-Mode for Inference
'''
class Text2Motion_RNN(nn.Module):
    def __init__(self,
                PATH=None,
                RUN_DEVICE=None,
                DEBUG_TIME=False,
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4
                ):
        super().__init__()
        self.n_layer = num_layers
        self.embed_dim = embed_dim
        self.RUN_DEVICE = RUN_DEVICE
        self.block_size = block_size
        self.num_vq = num_vq
        self.w = types.SimpleNamespace()
        w = torch.load(PATH, map_location=torch.device(RUN_DEVICE))['trans']
        for x in w.keys():
            w[x] = w[x].float()
            if '.time_' in x:
                w[x] = w[x].squeeze()
            if '.time_decay' in x:
                w[x] = -torch.exp(w[x])
            if DEBUG_TIME and '.time_' in x:
                print(x, w[x].squeeze().cpu().numpy())

            xx = x.split('.')
            here = self.w # 浅拷贝，here是self.w的引用
            for i in range(len(xx)):
                if xx[i].isdigit():
                    ii = int(xx[i])
                    if ii not in here:
                        here[ii] = types.SimpleNamespace()
                    here = here[ii]
                else:
                    if i == len(xx) - 1:
                        setattr(here, xx[i], w[x]) # 将对应权重存在最后一个命名空间中，即w['trans_base.blocks.0.att.time_decay']的权重存在time_decay的命名空间中
                    elif not hasattr(here, xx[i]):
                        if xx[i+1].isdigit():
                            setattr(here, xx[i], {})
                        else:
                            setattr(here, xx[i], types.SimpleNamespace())
                    here = getattr(here, xx[i]) # 保证命名空间的嵌套定义，如对于w['trans_base.blocks.0.att.time_decay']，此语句保证后续命名空间的创建都嵌套在trans_base中
        self.clear()
        self.eval()

    def clear(self):
        self.xx = {}
        self.aa = {}
        self.bb = {}
        self.pp = {}
        self.hk = None

    def save(self, target):
        target.xx = copy.deepcopy(self.xx)
        target.aa = copy.deepcopy(self.aa)
        target.bb = copy.deepcopy(self.bb)
        target.pp = copy.deepcopy(self.pp)
        target.hk = copy.deepcopy(self.hk)

    def load(self, target):
        self.xx = copy.deepcopy(target.xx)
        self.aa = copy.deepcopy(target.aa)
        self.bb = copy.deepcopy(target.bb)
        self.pp = copy.deepcopy(target.pp)
        self.hk = copy.deepcopy(target.hk)

    def ln(self, xx, w):
        return F.layer_norm(xx, (self.embed_dim,), weight=w.weight, bias=w.bias) 
    
    def linear(self, xx, w):
        if hasattr(w, 'bias'):
            return F.linear(xx, w.weight, w.bias)
        else :
            return F.linear(xx, w.weight)
        
    def FF(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.embed_dim, device=self.RUN_DEVICE)
        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)
        self.xx[name] = xx

        # r = torch.sigmoid(w.receptance.weight @ xr)
        # k = torch.square(torch.relu(w.key.weight @ xk))
        # kv = w.value.weight @ k
        r = torch.sigmoid(self.linear(xr, w.receptance))
        k = torch.square(torch.relu(self.linear(xk, w.key)))
        kv = self.linear(k, w.value)

        return r * kv
    
    def SA(self, xx, w, name):
        if name not in self.xx:
            self.xx[name] = torch.zeros(self.embed_dim, device=self.RUN_DEVICE)
            self.aa[name] = torch.zeros(self.embed_dim, device=self.RUN_DEVICE)
            self.bb[name] = torch.zeros(self.embed_dim, device=self.RUN_DEVICE)
            self.pp[name] = torch.zeros(self.embed_dim, device=self.RUN_DEVICE) - 1e30

        xk = xx * w.time_mix_k + self.xx[name] * (1 - w.time_mix_k)
        xv = xx * w.time_mix_v + self.xx[name] * (1 - w.time_mix_v)
        xr = xx * w.time_mix_r + self.xx[name] * (1 - w.time_mix_r)
        self.xx[name] = xx # cache current token as history token

        #r = torch.sigmoid(w.receptance.weight @ xr)
        #k = w.key.weight @ xk
        #v = w.value.weight @ xv
        r = torch.sigmoid(self.linear(xr, w.receptance))
        k = self.linear(xk, w.key)
        v = self.linear(xv, w.value)

        '''
        Cal the current wkv
        '''
        pp = self.pp[name]
        aa = self.aa[name]
        bb = self.bb[name]
        ww = w.time_first + k # time_first is corresponed to the u, ww is the base weight for current token
        q = torch.maximum(pp, ww) # for numerical safe
        e1 = torch.exp(pp - q)
        e2 = torch.exp(ww - q)
        a = e1 * aa + e2 * v # the numerator of wkv
        b = e1 * bb + e2 # the denominator of wkv

        '''
        Update the history state
        '''
        ww = pp + w.time_decay # time_decay is corresponed to the w
        q = torch.maximum(ww, k)
        e1 = torch.exp(ww - q)
        e2 = torch.exp(k - q)
        self.aa[name] = e1 * aa + e2 * v
        self.bb[name] = e1 * bb + e2
        self.pp[name] = q

        rwkv = r * a / b
        #return w.output.weight @ rwkv
        return self.linear(rwkv, w.output)
    
    def get_block_size(self):
        return self.block_size
    
    def forward(self, cur_token):
        w = self.w
        for i in range(self.n_layer):
            if i == 0:
                x = self.ln(cur_token, w.blocks[0].ln0)
            x = x + self.SA(self.ln(x, w.blocks[i].ln1), w.blocks[i].att, f'att.{i}')
            x = x + self.FF(self.ln(x, w.blocks[i].ln2), w.blocks[i].ffn, f'ffn.{i}')
    
        logits = self.linear(self.ln(x, w.ln_f), w.head)
        return logits
    
    def sample(self, clip_feature=None, if_categorial=False):
        # if clip_feature is not None:
        #     text_embdding = self.linear(clip_feature, self.w.cond_emb).unsqueeze(1) # (1, clip_dim) -> (1, 1, rwkv_dim)
        # else:
        #     text_embdding = self.w.un_cond.unsqueeze(0).unsqueeze(1)
        text_embdding = self.w.un_cond.unsqueeze(0).unsqueeze(1)
        insert = torch.randint(10, self.block_size-30, (1,)).to(text_embdding.device)
        for k in range(self.block_size): # 最大生成token长度
            if k == 0:
                x = text_embdding
            elif k == insert:
                self.clear()
                x = self.linear(clip_feature, self.w.cond_emb).unsqueeze(1) # (1, clip_dim) -> (1, 1, rwkv_dim)
            else:
                x = token_embedding
    
            logits = self.forward(x)
            # logits[:, :, 512] = -1e6
            probs = F.softmax(logits, dim=-1)

            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample() # input: (*, d), output: (*) indexs which sampled based on distribution d
                # import ipdb; ipdb.set_trace()
                
                if idx[0] == self.num_vq:
                    token_embedding = self.w.un_cond.unsqueeze(0).unsqueeze(1)
                    continue
                    # break
            else:
                _, idx = torch.topk(probs, k=1, dim=-1) # (1, 1, 1)
                idx = idx.flatten(1) # (1, 1, 1) -> (1, 1)
                if idx[0] == self.num_vq: # 对于<end> token，不将其加入输入序列。
                    break
            token_embedding = F.embedding(idx, self.w.tok_emb.weight) # the iput of next generation
            # append to the sequence and continue
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)
            
            if k == self.block_size - 1:
                return xs[:,:-1]
        
        return xs


def RWKV_Init(model, code_size, embd_dim):  # fancy initialization of all lin & emb layer in the model
    print("\n[--> first run, init model params (very slow for large models) <--]")
    print("[so you shall only do it for 1 single GPU and save the checkpt and load it when using multiple GPU]\n")

    for mm in model.modules():
        if "RecursiveScriptModule" in str(type(mm)):
            if mm.original_name not in ["Linear"]:
                continue
            ww = None
            for name, param in mm.named_parameters():
                if name == "weight":
                    ww = param
        else:
            m = mm
            if not isinstance(m, (nn.Linear, nn.Embedding)):
                continue
            ww = m.weight
        with torch.no_grad():
            name = "[unknown weight]"
            for name, parameter in model.named_parameters():  # find the name of the weight
                if id(ww) == id(parameter):
                    break

            shape = ww.shape
            gain = 1.0
            scale = 1.0  # extra scale for gain

            if isinstance(m, nn.Embedding):
                gain = math.sqrt(max(shape[0], shape[1]))
                if shape[0] == code_size + 2 and shape[1] == embd_dim:  # token emb?
                    scale = 1e-4
                else:
                    scale = 0

            if isinstance(m, nn.Linear):
                if shape[0] > shape[1]:
                    gain = math.sqrt(shape[0] / shape[1])
                if shape[0] == code_size + 1 and shape[1] == embd_dim:  # final projection?
                    scale = 0.5

            if hasattr(m, "scale_init"):
                scale = m.scale_init

            # print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {name}")

            gain *= scale
            if scale == -999:
                nn.init.eye_(ww)
            elif gain == 0:
                # zero init is great for some RWKV matrices
                nn.init.zeros_(ww)
            elif gain > 0:
                nn.init.orthogonal_(ww, gain=gain)
            else:
                nn.init.normal_(ww, mean=0.0, std=-scale)

'''
RWKV-v4 with GPT-Mode for Training
'''
class Text2Motion_Transformer(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        self.block_size = block_size
        self.num_vq = num_vq
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        with torch.no_grad():
            un_cond = nn.Parameter(torch.zeros(embed_dim))
            nn.init.normal_(un_cond, mean=0.0, std=1e-5)
        self.un_cond = un_cond
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.blocks = nn.Sequential(*[Block(num_layers, i, embed_dim) for i in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        try:
            RWKV_Init(self, num_vq, embed_dim) 
        except:
            pass
        # self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.01)
        if isinstance(module, (nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=1e-5)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature):
        if len(idxs) == 0: # for val
            # 固定随机种子
            torch.manual_seed(42)
            rand_value = torch.rand(1).item()
            if rand_value >= 0.5:
                token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
            else:
                token_embeddings = self.un_cond.unsqueeze(0).unsqueeze(1)
        else:
            b, t = idxs.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            mix_embeddings = self.cond_emb(clip_feature)
            mix_embeddings[0:b//2] = self.un_cond
            token_embeddings = self.tok_emb(idxs)
            token_embeddings = torch.cat([mix_embeddings.unsqueeze(1), token_embeddings], dim=1)
            
        x = self.blocks(token_embeddings)
        logits = self.head(self.ln_f(x)) 
        return logits # (B, N, 513)

    def sample(self, clip_feature, if_categorial=False):
        
        for k in range(self.block_size): # 最大生成token长度
            if k == 0:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature)
            logits = logits[:, -1, :] # (B, D)
            # list_logits.append(logits)
            probs = F.softmax(logits, dim=-1)
            # list_probs.append(probs)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample().unsqueeze(-1) # (B)
                if idx[0] == self.num_vq:
                    break
            else:
                _, idx = torch.topk(probs, k=1, dim=-1) # (B, 1)
                if idx[0] == self.num_vq: # 对于<end> token，不将其加入预测序列
                    break

            # GPT-Mode不缓存state，每次基于输入重新计算，与无KV Cache的解码过程相同
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)

            if k == self.block_size - 1:
                # with open('/home/adam/T2M-GPT/GPT-Mode/GPT_bat_probs.txt', 'a') as file:
                #     for i in range(len(list_probs)):
                #         file.write(f'pred_{i}\'s probs:{list_probs[i].cpu().numpy()}\n')
                # with open('/home/adam/T2M-GPT/GPT-Mode/GPT_bat_logits.txt', 'a') as file:
                #     for i in range(len(list_logits)):
                #         file.write(f'pred_{i}\'s logits:{list_logits[i].cpu().numpy()}\n')
                return xs[:, :-1]
            
        # with open('/home/adam/T2M-GPT/GPT-Mode/GPT_bat_probs.txt', 'a') as file:
        #     for i in range(len(list_probs)):
        #         file.write(f'pred_{i}\'s probs:{list_probs[i].cpu().numpy()}\n')
        # with open('/home/adam/T2M-GPT/GPT-Mode/GPT_bat_logits.txt', 'a') as file:
        #     for i in range(len(list_logits)):
        #         file.write(f'pred_{i}\'s logits:{list_logits[i].cpu().numpy()}\n')
        return xs
    
class RWKV_TimeMix(torch.jit.ScriptModule):
    def __init__(self, num_layers, layer_id, embed_dim):
        super().__init__()
        self.layer_id = layer_id
        #self.ctx_len = config.ctx_len
        self.n_embd = embed_dim
        attn_sz = embed_dim

        with torch.no_grad(): # fancy init
            ratio_0_to_1 = (layer_id / (num_layers - 1)) # 0 to 1
            ratio_1_to_almost0 = (1.0 - (layer_id / num_layers)) # 1 to ~0
            
            # fancy time_decay
            decay_speed = torch.ones(attn_sz)
            for h in range(attn_sz):
                decay_speed[h] = -5 + 8 * (h / (attn_sz-1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = (torch.tensor([(i+1)%3 - 1 for i in range(attn_sz)]) * 0.5)
            self.time_first = nn.Parameter(torch.ones(attn_sz) * math.log(0.3) + zigzag)
            
            # fancy time_mix
            x = torch.ones(1, 1, embed_dim)
            for i in range(embed_dim):
                x[0, 0, i] = i / embed_dim
            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(x, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(x, 0.5 * ratio_1_to_almost0))
            
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(embed_dim, attn_sz, bias=False)
        self.value = nn.Linear(embed_dim, attn_sz, bias=False)
        self.receptance = nn.Linear(embed_dim, attn_sz, bias=False)

        self.output = nn.Linear(attn_sz, embed_dim, bias=False)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    @torch.jit.script_method
    def jit_func(self, x):

        # Mix x with the previous timestep to produce xk, xv, xr
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        # Use xk, xv, xr to produce k, v, r
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        sr = torch.sigmoid(r)

        return sr, k, v
    def forward(self, x):
        B, T, C = x.size() # x = (Batch,Time,Channel)

        sr, k, v = self.jit_func(x)

        rwkv = sr * RUN_CUDA(B, T, C, self.time_decay, self.time_first, k, v)
        rwkv = self.output(rwkv)
        return rwkv
    
class RWKV_ChannelMix(torch.jit.ScriptModule):
    def __init__(self, num_layers, layer_id, embed_dim):
        super().__init__()
        self.layer_id = layer_id

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad(): # fancy init of time_mix
            ratio_1_to_almost0 = (1.0 - (layer_id / num_layers)) # 1 to ~0

            x = torch.ones(1, 1, embed_dim)
            for i in range(embed_dim):
                x[0, 0, i] = i / embed_dim

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))

        hidden_sz = 4 * embed_dim
        self.key = nn.Linear(embed_dim, hidden_sz, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(hidden_sz, embed_dim, bias=False)

        self.value.scale_init = 0 # 为Linear对象添加缩放属性
        self.receptance.scale_init = 0
    @torch.jit.script_method
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

        k = self.key(xk)
        k = torch.square(torch.relu(k))
        kv = self.value(k)

        rkv = torch.sigmoid(self.receptance(xr)) * kv
        return rkv

class Block(nn.Module):
    def __init__(self,
                 num_layers, 
                 layer_id,
                embed_dim=512,  
                #n_head=8, 
                ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(embed_dim)

        self.att = RWKV_TimeMix(num_layers, layer_id, embed_dim)
        self.ffn = RWKV_ChannelMix(num_layers, layer_id, embed_dim)
    def forward(self, x):
        if self.layer_id == 0:
            x = self.ln0(x)        
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# '''
# Transformer Decoder
# '''
# class Text2Motion_Transformer(nn.Module):

#     def __init__(self, 
#                 num_vq=1024, 
#                 embed_dim=512, 
#                 clip_dim=512, 
#                 block_size=16, 
#                 num_layers=2, 
#                 n_head=8, 
#                 drop_out_rate=0.1, 
#                 fc_rate=4):
#         super().__init__()
#         self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
#         self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
#         self.block_size = block_size
#         self.num_vq = num_vq

#     def get_block_size(self):
#         return self.block_size

#     def forward(self, idxs, clip_feature):
#         feat = self.trans_base(idxs, clip_feature)
#         logits = self.trans_head(feat)
#         return logits

#     def sample(self, clip_feature, if_categorial=False):
#         for k in range(self.block_size): # 最大生成token长度
#             if k == 0:
#                 x = []
#             else:
#                 x = xs
#             logits = self.forward(x, clip_feature)
#             logits = logits[:, -1, :]
#             probs = F.softmax(logits, dim=-1)
#             if if_categorial:
#                 dist = Categorical(probs)
#                 idx = dist.sample()
#                 # if idx == self.num_vq:
#                 #     break
#                 idx = idx.unsqueeze(-1)
#             else:
#                 _, idx = torch.topk(probs, k=1, dim=-1)
#                 if idx[0] == self.num_vq: # 如果生成了<end> token，则提前结束生成。
#                     idx = torch.randint(0, self.num_vq, (1,1)).to(probs.device) # for inference
#                     # break
#             # append to the sequence and continue
#             if k == 0:
#                 xs = idx
#             else:
#                 xs = torch.cat((xs, idx), dim=1)
            
#             if k == self.block_size - 1:
#                 return xs[:, :-1]
#         return xs


# class CausalCrossConditionalSelfAttention(nn.Module):

#     def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
#         super().__init__()
#         assert embed_dim % 8 == 0
#         # key, query, value projections for all heads
#         self.key = nn.Linear(embed_dim, embed_dim)
#         self.query = nn.Linear(embed_dim, embed_dim)
#         self.value = nn.Linear(embed_dim, embed_dim)

#         self.attn_drop = nn.Dropout(drop_out_rate)
#         self.resid_drop = nn.Dropout(drop_out_rate)

#         self.proj = nn.Linear(embed_dim, embed_dim)
#         # causal mask to ensure that attention is only applied to the left in the input sequence
#         # 用于记录不需要计算梯度但要跟模型参数一起保存、加载或移动的变量
#         self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
#         self.n_head = n_head

#     def forward(self, x):
#         B, T, C = x.size() 

#         # calculate query, key, values for all heads in batch and move head forward to be the batch dim
#         k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
#         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
#         att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
#         att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
#         att = F.softmax(att, dim=-1)
#         att = self.attn_drop(att)
#         y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
#         y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

#         # output projection
#         y = self.resid_drop(self.proj(y))
#         return y

# class Block(nn.Module):

#     def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
#         super().__init__()
#         self.ln1 = nn.LayerNorm(embed_dim)
#         self.ln2 = nn.LayerNorm(embed_dim)
#         self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, fc_rate * embed_dim),
#             nn.GELU(),
#             nn.Linear(fc_rate * embed_dim, embed_dim),
#             nn.Dropout(drop_out_rate),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.ln1(x))
#         x = x + self.mlp(self.ln2(x))
#         return x

# class CrossCondTransBase(nn.Module):

#     def __init__(self, 
#                 num_vq=1024, 
#                 embed_dim=512, 
#                 clip_dim=512, 
#                 block_size=16, 
#                 num_layers=2, 
#                 n_head=8, 
#                 drop_out_rate=0.1, 
#                 fc_rate=4):
#         super().__init__()
#         self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
#         self.cond_emb = nn.Linear(clip_dim, embed_dim)
#         self.pos_embedding = nn.Embedding(block_size, embed_dim)
#         self.drop = nn.Dropout(drop_out_rate)
#         # transformer block
#         self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
#         self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

#         self.block_size = block_size

#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
    
#     def forward(self, idx, clip_feature):
#         if len(idx) == 0:
#             token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
#         else:
#             b, t = idx.size()
#             assert t <= self.block_size, "Cannot forward, model block size is exhausted."
#             # forward the Trans model
#             token_embeddings = self.tok_emb(idx)
#             token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
#         x = self.pos_embed(token_embeddings)
#         x = self.blocks(x)

#         return x


# class CrossCondTransHead(nn.Module):

#     def __init__(self, 
#                 num_vq=1024, 
#                 embed_dim=512, 
#                 block_size=16, 
#                 num_layers=2, 
#                 n_head=8, 
#                 drop_out_rate=0.1, 
#                 fc_rate=4):
#         super().__init__()

#         self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
#         self.ln_f = nn.LayerNorm(embed_dim)
#         self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
#         self.block_size = block_size

#         self.apply(self._init_weights)

#     def get_block_size(self):
#         return self.block_size

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(self, x):
#         x = self.blocks(x)
#         x = self.ln_f(x)
#         logits = self.head(x)
#         return logits

    


        

