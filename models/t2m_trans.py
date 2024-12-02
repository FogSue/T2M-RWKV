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
RWKV_CTXLEN = 51
DTYPE = torch.float32
load(name="wkv6", sources=["./models/cuda/wkv6_op.cpp", f"./models/cuda/wkv6_cuda.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={RWKV_CTXLEN}"])
            
class WKV_6(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, k, v, w, u):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            assert C % HEAD_SIZE == 0
            assert r.dtype == DTYPE
            assert k.dtype == DTYPE
            assert v.dtype == DTYPE
            assert w.dtype == DTYPE
            assert u.dtype == DTYPE
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u)
            y = torch.empty((B, T, C), device=r.device, dtype=DTYPE, memory_format=torch.contiguous_format)
            torch.ops.wkv6.forward(B, T, C, H, r, k, v, w, u, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == DTYPE
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
            torch.ops.wkv6.backward(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
            gu = torch.sum(gu, 0).view(H, C//H)
            return (gr, gk, gv, gw, gu)

def RUN_CUDA_RWKV6(r, k, v, w, u):
    return WKV_6.apply(r, k, v, w, u)

'''
RWKV-v6 with RNN-Mode for Inference
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

def init_weights(model, embed_dim, num_layers, num_vq, DTYPE):
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
                    
            if DTYPE == torch.half:
                m[n] = m[n].half()
            elif DTYPE == torch.bfloat16:
                m[n] = m[n].bfloat16()
                
            m[n] = m[n].cpu()       
        gc.collect()
        torch.cuda.empty_cache()
        return m
'''
RWKV-v6 with GPT-Mode for Training
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
                drp_rate=0.2,
                gate_free=False
                ):
        super().__init__()
        self.block_size = block_size
        self.num_vq = num_vq
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.un_cond = nn.Parameter(torch.empty(embed_dim))
        self.cond_emb = nn.Linear(clip_dim, embed_dim, bias=False)
        self.blocks = nn.ModuleList([Block(num_layers, i, embed_dim, fc_rate, drp_rate, gate_free) for i in range(num_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        # self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     # print(f"Initializing: {type(module).__name__}")  # 打印每个被初始化的模块
    #     if isinstance(module, (nn.Linear)):
    #         module.weight.data.normal_(mean=0.0, std=0.01)
    #     if isinstance(module, (nn.Embedding)):
    #         module.weight.data.normal_(mean=0.0, std=1e-5)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()       

    def forward(self, idxs, clip_feature):
        if len(idxs) == 0: # for val
            x = self.cond_emb(clip_feature).unsqueeze(1)                           
        else:
            b, t = idxs.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            mix_embeddings = self.cond_emb(clip_feature) # (B, D)
            mix_embeddings[0:b//2] = self.un_cond
            token_embeddings = self.tok_emb(idxs)
            x = torch.cat([mix_embeddings.unsqueeze(1), token_embeddings], dim=1)
                
        for block in self.blocks:
            x = block(x)
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
                    break
            # GPT-Mode不缓存state，每次基于输入重新计算，与无KV Cache的解码过程相同
            if k == 0:
                xs = idx
            else:
                xs = torch.cat((xs, idx), dim=1)

            if k == self.block_size - 1:
                return xs[:, :-1]
            
        return xs


class RWKV_Tmix_x060(MyModule):
    def __init__(self, embed_dim, num_layers, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = HEAD_SIZE
        self.n_head = embed_dim // self.head_size
        assert embed_dim % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (num_layers- 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, embed_dim)
            for i in range(embed_dim):
                ddd[0, 0, i] = i / embed_dim

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            self.time_maa_w1 = nn.Parameter(torch.zeros(embed_dim, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, embed_dim).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(embed_dim)
            for n in range(embed_dim):
                decay_speed[n] = -6 + 5 * (n / (embed_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,embed_dim))

            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(embed_dim, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, embed_dim).uniform_(-0.01, 0.01))

            tmp = torch.zeros(embed_dim)
            for n in range(embed_dim):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (embed_dim - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))
            
            D_GATE_LORA = 64
            self.gate_w1 = nn.Parameter(torch.empty(embed_dim, D_GATE_LORA).uniform_(-0.01, 0.01))
            self.gate_w2 = nn.Parameter(torch.zeros(D_GATE_LORA, embed_dim).uniform_(-0.01, 0.01))
            # self.gate = nn.Linear(embed_dim, embed_dim, bias=False)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, embed_dim, eps=(1e-5)*(HEAD_SIZE_DIVISOR**2))

    @MyFunction
    def forward(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        # g = F.silu(self.gate(xg))
        g = torch.tanh(xg @ self.gate_w1) @ self.gate_w2

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)

        x = x.view(B * T, C)        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x
    
class RWKV_Tmix_x060_Gate_Free(MyModule):
    def __init__(self, embed_dim, num_layers, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.head_size = HEAD_SIZE
        self.n_head = embed_dim // self.head_size
        assert embed_dim % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (num_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, embed_dim)
            for i in range(embed_dim):
                ddd[0, 0, i] = i / embed_dim 
            # shallow-big, deeper-small
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            D_MIX_LORA = 32
            self.time_maa_rkvw_w1 = nn.Parameter(torch.zeros(embed_dim, D_MIX_LORA*4))
            self.time_maa_rkvw_w2 = nn.Parameter(torch.zeros(4, D_MIX_LORA, embed_dim).uniform_(-0.01, 0.01))

            decay_speed = torch.ones(embed_dim)
            for n in range(embed_dim):
                decay_speed[n] = -6 + 5 * (n / (embed_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1) # deepr layer has higher decay_speed
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,embed_dim))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(embed_dim, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, embed_dim).uniform_(-0.01, 0.01))

            tmp = torch.zeros(embed_dim)
            for n in range(embed_dim):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (embed_dim - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size)) # shared across layers

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)

        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_x = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x
        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_rkvw_w1).view(B*T, 4, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_rkvw_w2).view(4, B, T, C)

        r, k, v, w = xxx.unbind(dim=0)
        r = x + xx * (self.time_maa_r + r)
        k = x + xx * (self.time_maa_k + k)
        v = x + xx * (self.time_maa_v + v)
        w = x + xx * (self.time_maa_w + w)
        
        r = self.receptance(r)
        k = self.key(k)
        v = self.value(v)
        w = self.time_decay + torch.tanh(w @ self.time_decay_w1) @ self.time_decay_w2
        k = k * (1-(-w.exp()).exp()) # for fp32
        # k = k * (1-(-w.float().exp()).exp()).to(dtype=torch.bfloat16) # for bf16
        
        x = RUN_CUDA_RWKV6(r, k, v, w, u=self.time_faaaa)
        x = self.ln_x(x)
        x = self.output(x)
        return x
class RWKV_CMix_x060(MyModule):
    def __init__(self, embed_dim, num_layers, layer_id, fc_rate):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / num_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, embed_dim)
            for i in range(embed_dim):
                ddd[0, 0, i] = i / embed_dim
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
        dim_ffn = embed_dim * fc_rate
        self.key = nn.Linear(embed_dim, dim_ffn, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(dim_ffn, embed_dim, bias=False)
    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class Block(nn.Module):
    def __init__(self,
                 num_layers, 
                 layer_id,
                 embed_dim,
                 fc_rate=4,
                 drp_rate=0.2,
                 gate_free = True
                ):
        super().__init__()
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(drp_rate)
        self.drop2 = nn.Dropout(drp_rate)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(embed_dim)

        if gate_free:
            self.att = RWKV_Tmix_x060_Gate_Free(embed_dim, num_layers, layer_id)
        else:
            self.att = RWKV_Tmix_x060(embed_dim, num_layers, layer_id)
        self.ffn = RWKV_CMix_x060(embed_dim, num_layers, layer_id, fc_rate)
        
    def forward(self, x):

        if self.layer_id == 0:
            x = self.ln0(x)

        x = self.drop1(x + self.att(self.ln1(x)))
        x = self.drop2(x + self.ffn(self.ln2(x)))

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

    


        

