import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32   # larger --> more memory, stable gradients, better parallelization
block_size = 8    # larger --> more context (better for complex syntactic structures), more compute
max_iters = 10000  # too big gives overfitting, too small gives underfitting (convergence)
eval_interval = 500   # frequent evals gives closer monitoring, but can slow down training
learning_rate = 1e-2  # too big gives divergence (fails to converge), too small gives local minima (AdamW helps)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

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

# model (very simple bigram language model)
class BigramLanguageModel(nn.Module):
  
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
  def forward(self, idx, target=None):
    # idx and targets are both (B,T) tensor of integers
    logits = self.token_embedding_table(idx)  # (B,T,C)
    
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
      logits, loss = model(idx)   # (B,T,C)
      logits = logits[:, -1, :]   # (B,C)
      probs = F.softmax(logits, dim=-1)   # (B,C)
      next_idx = torch.multinomial(probs, num_samples=1)    # (B,1)
      idx = torch.cat([idx, next_idx], dim=1)   # (B,T+1)
    return idx

# instantiate model
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)    

# training loop
for iter in range(max_iters):
  x, y = get_batch('train')   # sample a batch of data
  logits, loss = model(x, y)  # forward pass
  optimizer.zero_grad()       # zero out the gradients
  loss.backward()             # backward pass
  optimizer.step()            # update the weights
  
  # print loss on train and val sets after every few iterations
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f'Iter: {iter}, Train loss: {losses["train"]}, Val loss: {losses["val"]}')

# generate from the model
context = torch.zeros(1, 1, dtype=torch.long).to(device)
print(decode(m.generate(context, 1000)[0].tolist()))