import math, os, gc
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

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

from torch.utils.cpp_extension import load
# os.environ['RWKV_FLOAT_MODE'] = 'bf16'
HEAD_SIZE = 64 # don't change
HEAD_SIZE_DIVISOR = 8 # don't change
DTYPE = torch.float32
# RWKV_CTXLEN = 51

'''
For long context training, this chunk kernel improves efficiency, 
but it does not support validation unless CHUNK_LEN is set to 1.
'''
CHUNK_LEN = 51
flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
load(name="wind_backstepping", sources=[f'./models/cuda/wkv7_train_cuda.cu', './models/cuda/wkv7_train_op.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0
        assert all(i.dtype == DTYPE for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype = DTYPE, device = w.device)
        sa = torch.empty(B,T,H,C, dtype = DTYPE, device = w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.float32 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db

def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

load(name="wkv7", sources=["./models/cuda/wkv7_eval_op.cpp", f"./models/cuda/wkv7_eval_cuda.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert r.dtype == DTYPE
            assert w.dtype == DTYPE
            assert k.dtype == DTYPE
            assert v.dtype == DTYPE
            assert a.dtype == DTYPE
            assert b.dtype == DTYPE
            assert r.is_contiguous()
            assert w.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert a.is_contiguous()
            assert b.is_contiguous()
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
            return y

def RWKV7_OP(r, w, k, v, a, b):
    return WKV_7.apply(r, w, k, v, a, b)

'''
RWKV-v7 with RNN-Mode for Inference
'''
class RWKV_RNN(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eval() # set torch to inference mode
        
        w = torch.load(args.MODEL_NAME + '.pth', map_location='cpu')

        for k in w.keys():
            w[k] = w[k].float() # convert to f32 type
            if      '.time_' in k: w[k] = w[k].squeeze()
            if '.time_faaaa' in k: w[k] = w[k].unsqueeze(-1)

        self.n_head = w['blocks.0.att.time_faaaa'].shape[0]
        self.head_size = w['blocks.0.ln1.weight'].shape[0] // self.n_head
        
        self.w = types.SimpleNamespace() # set self.w from w
        self.w.blocks = {}
        for k in w.keys(): # example: "blocks.0.att.time_first" => self.w.blocks[0].att.time_first
            parts = k.split('.')
            last = parts.pop()
            here = self.w
            for p in parts:
                if p.isdigit():
                    p = int(p)
                    if p not in here: here[p] = types.SimpleNamespace()
                    here = here[p]
                else:
                    if not hasattr(here, p): setattr(here, p, types.SimpleNamespace())
                    here = getattr(here, p)
            setattr(here, last, w[k])

    def layer_norm(self, x, w):
        return F.layer_norm(x, (self.args.n_embd,), weight=w.weight, bias=w.bias)

    @MyFunction
    def channel_mixing(self, x, state, i:int, time_maa_k, time_maa_r, kw, vw, rw):
        i0 = (2+self.head_size)*i+0
        sx = state[i0] - x
        xk = x + sx * time_maa_k
        xr = x + sx * time_maa_r
        state[i0] = x
        r = torch.sigmoid(rw @ xr)
        k = torch.square(torch.relu(kw @ xk)) # square relu, primer paper
        return r * (vw @ k)

    @MyFunction
    def time_mixing(self, x, state, i:int, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, time_first, time_decay, kw, vw, rw, gw, ow, ln_w, ln_b):
        H = self.n_head
        S = self.head_size

        i1 = (2+S)*i+1
        sx = state[i1] - x
        state[i1] = x
        xxx = x + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + sx * (w_maa + mw)
        xk = x + sx * (k_maa + mk)
        xv = x + sx * (v_maa + mv)
        xr = x + sx * (r_maa + mr)
        xg = x + sx * (g_maa + mg)

        w = (time_decay + (torch.tanh(xw @ td_w1) @ td_w2).float()).view(H, S, 1)
        w = torch.exp(-torch.exp(w.float()))

        r = (rw @ xr).view(H, 1, S)
        k = (kw @ xk).view(H, S, 1)
        v = (vw @ xv).view(H, 1, S)
        g = F.silu(gw @ xg)

        s = state[(2+S)*i+2:(2+S)*(i+1), :].reshape(H, S, S)

        x = torch.zeros(H, S)
        a = k @ v
        x = r @ (time_first * a + s)
        s = a + w * s
    
        state[(2+S)*i+2:(2+S)*(i+1), :] = s.reshape(S, -1)
        x = x.flatten()

        x = F.group_norm(x.unsqueeze(0), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).squeeze(0) * g # same as gn(x/8, eps=1e-5)
        return ow @ x

    def forward(self, token, state=None):
        with torch.no_grad():
            if state == None:
                state = torch.zeros(self.args.n_layer * (2+self.head_size), self.args.n_embd)
            
            x = self.w.emb.weight[token]
            x = self.layer_norm(x, self.w.blocks[0].ln0)
            for i in range(self.args.n_layer):
                att = self.w.blocks[i].att
                x = x + self.time_mixing(self.layer_norm(x, self.w.blocks[i].ln1), state, i,
                    att.time_maa_x, att.time_maa_w, att.time_maa_k, att.time_maa_v, att.time_maa_r, att.time_maa_g, att.time_maa_w1, att.time_maa_w2,
                    att.time_decay_w1, att.time_decay_w2, att.time_faaaa, att.time_decay,
                    att.key.weight, att.value.weight, att.receptance.weight, att.gate.weight, att.output.weight,
                    att.ln_x.weight, att.ln_x.bias)
                ffn = self.w.blocks[i].ffn
                x = x + self.channel_mixing(self.layer_norm(x, self.w.blocks[i].ln2), state, i, 
                    ffn.time_maa_k, ffn.time_maa_r, 
                    ffn.key.weight, ffn.value.weight, ffn.receptance.weight)
            
            x = self.w.head.weight @ self.layer_norm(x, self.w.ln_out)
            return x.float(), state
    
    def sample(self, clip_feature=None, if_categorial=False):
        if clip_feature is not None:
            text_embdding = self.linear(clip_feature, self.w.cond_emb).unsqueeze(1) # (1, clip_dim) -> (1, 1, rwkv_dim)
        else:
            text_embdding = self.w.un_cond.unsqueeze(0).unsqueeze(1)
        # text_embdding = self.w.un_cond.unsqueeze(0).unsqueeze(1)
        # insert = torch.randint(1, self.block_size, (1,)).to(text_embdding.device)
        for k in range(self.block_size): # 最大生成token长度
            if k == 0:
                x = text_embdding
            # elif k == insert:
            #     self.clear()
            #     x = self.linear(clip_feature, self.w.cond_emb).unsqueeze(1) # (1, clip_dim) -> (1, 1, rwkv_dim)
            else:
                x = token_embedding
    
            logits = self.forward(x)
            # logits[:, :, 512] = -1e6
            probs = F.softmax(logits, dim=-1)

            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample() # input: (*, d), output: (*) indexs which sampled based on distribution d    
                if idx[0] == self.num_vq:
                    # token_embedding = self.w.un_cond.unsqueeze(0).unsqueeze(1)
                    # continue
                    break
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

def init_weights(model, embed_dim, num_layers, num_vq):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = model.state_dict()
        for n, p in model.state_dict().items():
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias'):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / num_layers
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
                print()
            elif n == "tok_emb.weight" or n == 'un_cond':
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if num_vq > embed_dim:
                    scale = 0.5 * math.sqrt(num_vq / embed_dim)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith('.weight') # should always be true
                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

                for kk in zero:
                    if kk in n:
                        scale = 0
                if "head_k." in n:
                    scale = 0.1
                if "head_q." in n:
                    scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")
                m[n] = torch.empty((shape[0], shape[1]), device=p.device)
                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)
            # if os.environ["RWKV_FLOAT_MODE"] == "fp16":
            #     m[n] = m[n].half()
            # elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
            #     m[n] = m[n].bfloat16()#to(dtype=torch.bfloat16)
            m[n] = m[n].cpu()       
        gc.collect()
        torch.cuda.empty_cache()
        return m
'''
RWKV-v7 with GPT-Mode for Training
'''
class Text2Motion_Transformer(nn.Module):
    def __init__(
                self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2,
                fc_rate=4,
                drp_rate=0.1,
                ):
        super().__init__()
        self.block_size = block_size
        self.num_vq = num_vq
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.un_cond = nn.Parameter(torch.empty(embed_dim).normal_(mean=0.0, std=0.02))
        self.cond_emb = nn.Linear(clip_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList([Block(num_layers, i, embed_dim, fc_rate, drp_rate) for i in range(num_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        
        self.tok_emb.weight.data.normal_(mean=0., std=0.02)
        self.cond_emb.weight.data.normal_(mean=0.0, std=0.02)
        self.head.weight.data.normal_(mean=0.0, std=0.02)
        
    def forward(self, idxs, clip_feature):
        if len(idxs) == 0: # for val
            assert self.training != True
            x = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idxs.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            mix_embeddings = self.cond_emb(clip_feature) # (B, D)
            mix_embeddings[0:b//2] = self.un_cond
            token_embeddings = self.tok_emb(idxs)
            x = torch.cat([mix_embeddings.unsqueeze(1), token_embeddings], dim=1)
                
            
        v0 = torch.empty_like(x)
        for block in self.blocks:
            x, v0 = block(x, v0)
        logits = self.head(self.ln_out(x)) 
        return logits # (B, N, 513)

    def sample(self, clip_feature, if_categorial=False):
        assert self.training == False        
        for k in range(self.block_size): # 最大生成token长度
            if k == 0:
                x = []
            else:
                x = xs
            logits = self.forward(x, clip_feature)
            logits = logits[:, -1, :] # (1, D)
            probs = F.softmax(logits, dim=-1)
            if if_categorial:
                dist = Categorical(probs)
                idx = dist.sample().unsqueeze(-1) # (1)
                if idx[0] == self.num_vq:
                    break
            else:
                _, idx = torch.topk(probs, k=1, dim=-1) # (1, 1)
                if idx[0] == self.num_vq: # 对于<end> token，不将其加入预测序列
                    # print(f"End in {k} step")
                    break

            # GPT-Mode不缓存state，每次基于输入重新计算，与无KV Cache的解码过程相同
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)

            if k == self.block_size - 1:
                return xs[:, :-1]
            
        return xs


class RWKV_Tmix_x070(MyModule):
    def __init__(self, embed_dim, num_layers, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = HEAD_SIZE
        self.n_head = embed_dim // self.head_size
        assert embed_dim % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = embed_dim

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (num_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1))
            self.time_maa_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = 64 # dim 64 for emb 768, change it for smaller/larger models
            self.time_decay_w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = 32 # dim 32 for emb 768, change it for smaller/larger models
            self.time_aaa_w1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.time_aaa_w2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.time_aaaaa = nn.Parameter(torch.zeros(1,1,C))

            D_MV_LORA = 32 # dim 32 for emb 768, change it for smaller/larger models
            self.mv_w1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.mv_w2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.time_misc_v = nn.Parameter(torch.zeros(1,1,C)+1.0)

            D_GATE_LORA = 128 # dim 128 for emb 768, change it for smaller/larger models
            # Note: for some data, you can reduce gate lora dimension (such as to 32), or even remove it
            self.gate_w1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.gate_w2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.time_misc_kkk = nn.Parameter(torch.ones(1,1,C)*0.85)
            self.time_misc_a = nn.Parameter(torch.ones(1,1,C))

            self.time_faaaa = nn.Parameter(torch.zeros(1,1,H,N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=(1e-5)*(HEAD_SIZE_DIVISOR**2))

            # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.uniform_()

    def forward(self, x, v0):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.time_maa_r
        xw = x + xx * self.time_maa_w
        xk = x + xx * self.time_maa_k
        xv = x + xx * self.time_maa_v
        xa = x + xx * self.time_maa_a
        xg = x + xx * self.time_maa_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v0 = v
        else:
            v = v + (v0 - v) * torch.sigmoid(self.time_misc_v + (xv @ self.mv_w1) @ self.mv_w2)
        a = torch.sigmoid(self.time_aaaaa + (xa @ self.time_aaa_w1) @ self.time_aaa_w2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.gate_w1) @ self.gate_w2

        kk = k * self.time_misc_kkk
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.time_misc_a)
        
        if self.training:
            x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        else:
            x = RWKV7_OP(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.time_faaaa).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v0
    
class RWKV_CMix_x070(MyModule):
    def __init__(self, embed_dim, num_layers, layer_id, fc_rate):
        super().__init__()
        self.args = embed_dim
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, embed_dim)
            for i in range(embed_dim):
                ddd[0, 0, i] = i / embed_dim
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(embed_dim, embed_dim*fc_rate, bias=False)
        self.value = nn.Linear(embed_dim*fc_rate, embed_dim, bias=False)

        # !!! initialize if you are using RWKV_Tmix_x070 in your code !!!
        self.key.weight.data.uniform_(-0.5/(embed_dim**0.5), 0.5/(embed_dim**0.5))
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        
        k = x + xx * self.time_maa_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)

class Block(nn.Module):
    def __init__(self,
                 num_layers, 
                 layer_id,
                 embed_dim,
                 fc_rate=4,
                 drp_rate=0.1,
                ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(drp_rate)
        self.drop2 = nn.Dropout(drp_rate)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(embed_dim)

        self.att = RWKV_Tmix_x070(embed_dim, num_layers, layer_id)
        self.ffn = RWKV_CMix_x070(embed_dim, num_layers, layer_id, fc_rate)
        
    def forward(self, x, v0):
        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v0 = self.att(self.ln1(x), v0)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v0


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

    


        

