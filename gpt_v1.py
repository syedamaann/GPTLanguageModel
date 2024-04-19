import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64   # larger --> more memory, stable gradients, better parallelization
block_size = 256    # larger --> more context (better for complex syntactic structures), more compute
max_iters = 5000  # too big gives overfitting, too small gives underfitting (convergence)
eval_interval = 500   # frequent evals gives closer monitoring, but can slow down training
learning_rate = 3e-4  # too big gives divergence (fails to converge), too small gives local minima (AdamW helps)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.3

# data preprocessing
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt 

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8' ) as f:
    text = f.read()
    
# create vocabulary (unique characters that occur in this text)
vocab = sorted(list(set(text)))
vocab_size = len(vocab)

# create mapping from characters to integers
stoi = {c:i for i,c in enumerate(vocab)}
itos = {i:c for i,c in enumerate(vocab)}
encode = lambda x: [stoi[c] for c in x]   # encoder: takes a string, output a list of integers
decode = lambda x: ''.join([itos[i] for i in x])  # decoder: takes a list of integers and outputs a string

# train and val split
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.9*len(data))]
val_data = data[int(0.9*len(data)):]

# create dataloaders
def get_batch(split):
    data = train_data if split == 'train' else val_data             
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y
  
# estimate loss
@torch.no_grad()    # no need to compute gradients
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      x, y = get_batch(split)
      logits, loss = model(x,y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train()
  return out

class ParallelMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.n_head = n_head
        self.head_size = head_size
        self.key_query_value = nn.Linear(n_embd, 3 * head_size * n_head, bias=False)
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        # Register buffer for the triangular mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.key_query_value(x)
        q, k, v = qkv.split(self.head_size * self.n_head, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # Compute attention scores ("affinities")
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        # Apply the mask, making sure to expand it to cover all heads and batch size
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # Perform the weighted aggregation of the values
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_size)

        # Final projection
        out = self.proj(out)
        return out


# feedforward layer
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = ParallelMultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # communication (self-attention)
        x = x + self.ffwd(self.ln2(x))  # computation (feedforward)
        return x
      
# GPT model
class GPTLanguageModel(nn.Module):
  
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)       # token embeddings
    self.position_embedding_table = nn.Embedding(block_size, n_embd)    # positional embeddings
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)                        # linear layer
    self.apply(self._init_weights)

  # initialize weights
  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
  def forward(self, idx, target=None):
    # idx and targets are both (B,T) tensor of integers
    B,T = idx.shape
    token_emb = self.token_embedding_table(idx)  # (B,T,C)
    pos_embd = self.position_embedding_table(torch.arange(T).to(device))  # (T,C)
    x = token_emb + pos_embd    # (B,T,C)
    x = self.blocks(x)          # (B,T,C)
    x = self.ln_f(x)            # (B,T,C)
    logits = self.lm_head(x)    # (B,T, vocab_size)
    
    if target is None:
      loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T,C)   # why because cross_entropy expects (N,C) input
      target = target.view(B*T)     # why because cross_entropy expects (N,) target
      loss = F.cross_entropy(logits, target)    
      
    return logits, loss
  
  def generate(self, idx, max_new_tokens):
    # idx is a (B,T) tensor of integers
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] 
      logits, loss = self(idx_cond)   # (B,T,C)
      logits = logits[:, -1, :]   # (B,C)
      probs = F.softmax(logits, dim=-1)   # (B,C)
      next_idx = torch.multinomial(probs, num_samples=1)    # (B,1)
      idx = torch.cat([idx, next_idx], dim=1)   # (B,T+1)
    return idx

# instantiate model
model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)    

# training loop
for iter in range(max_iters):
  x, y = get_batch('train')   # sample a batch of data
  logits, loss = model(x, y)  # forward pass
  optimizer.zero_grad(set_to_none=True)       # zero out the gradients
  loss.backward()             # backward pass
  optimizer.step()            # update the weights
  
  # print loss on train and val sets after every few iterations
  if iter % eval_interval == 0 or iter == max_iters - 1:
    losses = estimate_loss()
    print(f'Iter: {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

# generate from the model
context = torch.zeros(1, 1, dtype=torch.long).to(device)
print(decode(m.generate(context, 1000)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))