import torch
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim

# Hyperparameters
batch_size=32
block_size=8 # context windows
learning_rate=1e-3
n_epoch=10000
device='cuda' if torch.cuda.is_available() else 'cpu'
eval_iter=10
vocab_size=65
n_embd=32
head_size=16




# seed 
torch.manual_seed(1337)

# Reading data 
with open('data/tinyshakespear.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars=sorted(list(set(text)))
vocab_size=len(chars)
# transforming char to index and vice versa
char_to_index={c:i for i,c in enumerate(chars)}
index_to_char={i:c for i,c in enumerate(chars)}


# encoder/tokenizer
tokenize=lambda text:[char_to_index[c] for c in text]

# split train/validation 
data=tokenize(text)
N=len(data)
data=torch.tensor(data)
train=data[:int(N*0.9)]
validation=data[:int(N*0.9):]

def get_batch(split):
    data=train if split=="train" else validation
    ix=torch.randint(0, len(data)-block_size, (batch_size,))
    X=torch.stack([data[i:i+block_size] for i in ix ])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix ])
    x,y=X.to(device),y.to(device)
    return X,y


@torch.no_grad
def estimate_loss():
    out={} # put training loss and validation loss
    model.eval() # setting model for eval
    for split in ["train","validation"]:
        losses=torch.zeros(eval_iter)
        for k in range(eval_iter):
            xb,yb=get_batch(split)
            loss,_=model(xb,yb)
            losses[k]=loss

        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.q=nn.Linear(n_embd,head_size,bias=False)
        self.k=nn.Linear(n_embd,head_size,False)
        self.v=nn.Linear(n_embd,head_size,False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))


    def forward(self,x):
        B,T,C=x.shape
        wei=self.q(x) @ self.k(x).transpose(-2,-1) * (C**-0.5) # BTH@BHT --->BTT
        # add mask to stop from seing future tokens
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1) # BTT
        # value
        out =wei@self.v(x) # B,T,T @ B,T,H --> B,T,H

        return out

        

# Bigram model
class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.token_embed=nn.Embedding(vocab_size,n_embd)
        self.pos=nn.Embedding(block_size,n_embd)
        self.head=Head(head_size=head_size)
        self.lm_head=nn.Linear(head_size,vocab_size)

    def forward(self,input,target=None):
        B,T=input.shape
        embedding=self.token_embed(input) # B,T ---> B,T,n_embd
        pos_embd=self.pos(torch.arange(T))#B,T,n_embd
        input=embedding+pos_embd # encoded the position (B,T,n_embd) 
        h=self.head(input) #B,T,Head_size
        logits=self.lm_head(h)

        B,T,C=logits.shape
        if target!=None:
            logits=logits.view(B*T,C) # B*T,C
            target=target.view(B*T) # B*T
            loss=F.cross_entropy(logits,target)
            return loss,logits
        else:
            return logits
    
    def generate(self,ix,max_len=100):
        for _ in range(max_len):
            idx_cond=ix[:,-block_size:]
            logits=self(idx_cond) # B,T--->B,T,C (logits)
            proba=F.softmax(logits[:,-1,:],dim=1) # B,C (last element)
            idx_next = torch.multinomial(proba, num_samples=1) # B,1
            ix=torch.concat([ix,idx_next],dim=1) # B,T concat B,1 ---> B,T+1
            
        return ix

         
model=BigramModel()
model=model.to(device)


#----------------------------------
#    training 
#----------------------------------

# Define the optimizer
adam = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(n_epoch):
    xb,yb=get_batch("train")
    # zero the grad 
    adam.zero_grad()
    # make prediction
    loss,logits=model(xb,yb)
    # estimate training and validation losses
    out=estimate_loss()
    # Print stats
    if epoch%10==0: print(f"Epoch {epoch}: traing loss: {out['train']} validation loss: {out["validation"]}")
    # backprop
    loss.backward()
    # step
    adam.step()


# generate
context=torch.tensor([[5]],dtype=torch.long,device=device)
generated_tokens = model.generate(context,max_len=1000)
print("".join([index_to_char[gt.item()] for gt in generated_tokens[0]])) # [O] for batch