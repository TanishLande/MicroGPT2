import torch
import torch.nn as nn
from dataclasses import dataclass
import math
import torch.nn.functional as F
import tiktoken
import inspect
import torch.distributed as dist
import os

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.SCALE_GPT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # manual implementation of attention
        # this materializes the large (T,T) matrix for all the queries and keys

        #  normal attention
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # flash attention 
        y = F.scaled_dot_product_attention(q, k, v ,is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(config.n_embd * 4 , config.n_embd)
        self.c_proj.SCALE_GPT = 1

    def forward(self,x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPT2config:
    block_size: int  = 1024
    vocab_size: int = 50257
    n_layer:  int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing trick: tie the input and output embeddings
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_GPT') and module.SCALE_GPT == 1:
                std *= (2* self.config.n_layer) ** -0.5  # scale for GPT
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # initialize embeddings with a normal distribution
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device )
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        for block in self.transformer.h:
            x = block(x)        
        
        x = self.transformer.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPT2config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# output

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ====================
class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, world_size=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.world_size = world_size

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens  = torch.tensor(tokens)
        print(f"loaded {len(tokens)} tokens")
        print(f"1 epchs = {len(self.tokens) // (B * T)} batches")

        self.current_position =  self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T =  self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # shape: [B, T] : input tokens
        y = buf[1:].view(B, T)  # shape: [B, T] : output tokens (shifted by 1)
        self.current_position += B * T * self.process_rank
        if self.current_position + (B * T * self.process_rank + 1 ) >= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x , y


# --------------------------------------------------
import time

# DDP : paraller processing of multiple GPUs
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")



torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T * ddp_world_size)  == 0, "make sure that total_batch_size is divisible by B * T * ddp_world_size"
grad_accumulation_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total_batch_size: {total_batch_size}, grad_accumulation_steps: {grad_accumulation_steps}")
    print(f"grad_accumulation_steps: {grad_accumulation_steps}, B: {B}, T: {T}")


train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, world_size=ddp_world_size)

torch.set_float32_matmul_precision('high')

model = GPT(GPT2config(vocab_size=50304))
model.to(device)
model = torch.compile(model, backend="eager")
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # get the raw model for optimizer

max_lr  = 6e-4
min_lr  = max_lr * 0.1
warumup_steps = 10
max_steps = 50
def get_lr(it):
    if it  < warumup_steps:
        return max_lr * (it + 1) / warumup_steps

    if it > max_steps :
        return min_lr 

    decay_ratio = (it - warumup_steps) / (max_steps - warumup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
    
train_loader = DataLoaderLite(B=4, T=32)

# optimizer 
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device)

for step in range(max_steps):
    t0 = time.time()
    for micro_step in range(grad_accumulation_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)  # move to GPU if
        optimizer.zero_grad()
        loss_accum = 0.0
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accumulation_steps  # scale the loss for gradient accumulation
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accumulation_steps - 1)  # sync gradients only on the last micro-step
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)  # average the loss across all processes
    norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # making the gradients stable

    lr  = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    torch.cuda.synchronize()  # wait for all GPU operations to finish
    t1 = time.time()
    dt = (t1 - t0)*1000
    tokens_per_Sec = (train_loader.B * train_loader.T) / (t1 - t0)  # tokens per second
    tokens_proccessed = train_loader.B * train_loader.T * grad_accumulation_steps * ddp_world_size
    if master_process:
        print(f"Step {step+1}, Loss: {loss_accum.item():.4f}, lr: {lr:.4e} ,  norm: {norm:.4f} , dt: {dt:.2f} ms, tokens/sec: {tokens_per_Sec:.2f}")

if ddp:
    destroy_process_group()  # clean up DDP process group

import sys; sys.exit(0)


# enc = tiktoken.get_encoding("gpt2")
# prompt = "What is meaning of life?"
# input_tokens = enc.encode(prompt)
# input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to('cuda')  # shape: [1, T]

# num_return_sequences = 5
# max_length = 30

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# model = GPT.from_pretrained('gpt2')
# model.eval().to('cuda')

# for i in range(num_return_sequences):
#     x = input_tensor.clone()
#     while x.size(1) < max_length:
#         with torch.no_grad():
#             logits = model(x)
#             logits = logits[:, -1, :]
#             probs = F.softmax(logits, dim=-1)
#             top_k, topk_indices = torch.topk(probs, 50, dim=-1)
#             ix = torch.multinomial(top_k, 1)
#             xcol = torch.gather(topk_indices, dim=-1, index=ix)
#             x = torch.cat((x, xcol), dim=1)

#     tokens = x[0, :max_length].tolist()
#     decoded = enc.decode(tokens)
#     print(f"Generated sequence {i + 1}: {decoded}")