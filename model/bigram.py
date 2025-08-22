import torch
import torch.nn as nn 
from torch.nn import functional as F
import torch.optim as optim

# Hyperparameters
batch_size=4
block_size=8 # context windows
learning_rate=1e-1
n_epoch=10000
device='cuda' if torch.cuda.is_available() else 'cpu'




# seed 
torch.manual_seed(1337)

# Reading data 
with open('../data/tinyshakespear.txt', 'r', encoding='utf-8') as f:
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

def get_batch(data):
    ix=torch.randint(0, len(data)-block_size, (batch_size,))
    X=torch.stack([data[i:i+block_size] for i in ix ])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix ])
    return X,y


# Bigram model
class BigramModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embed=nn.Embedding(vocab_size,vocab_size)

    def forward(self,input,target=None):

        logits=self.token_embed(input) # B,T ---> B,T,C
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
            logits=self(ix) # B,T--->B,T,C (logits)
            proba=F.softmax(logits[:,-1,:],dim=1) # B,C (last element)
            idx_next = torch.multinomial(proba, num_samples=1) # B,1
            ix=torch.concat([ix,idx_next],dim=1) # B,T concat B,1 ---> B,T+1
            
        return ix

         
bm=BigramModel(vocab_size)


# Define the optimizer
adam = optim.Adam(bm.parameters(), lr=learning_rate)

for epoch in range(n_epoch):
    xb,yb=get_batch(train)
    # zero the grad 
    adam.zero_grad()
    # make prediction
    loss,logits=bm(xb,yb)
    # Print stats
    if epoch%100==0: print(f"Epoch {epoch}: traing loss: {loss}")
    # backprop
    loss.backward()
    # step
    adam.step()


input=torch.tensor([[5]])
generated_tokens = bm.generate(input,max_len=1000)
print("".join([index_to_char[gt.item()] for gt in generated_tokens[0]])) # [O] for batch