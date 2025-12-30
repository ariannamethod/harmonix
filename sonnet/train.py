
import torch
import torch.nn as nn
from torch.nn import functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
max_steps = 9000
n_embd = 128
dropout = 0.1
n_head = 4
n_layer = 4
block_size = 64
batch_size = 16

# For reproducibility.
# torch.manual_seed(1337)

#Sets the the open TinyShakeSpeare function as f and then assigns it's contents to the variable 'text'.
with open('TinyShakeSpeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#'Set()' removes all duplicate characters and 'Sorted()' sorts the characters in an ascending order.
chars=(sorted(set(text)))
vocab_size=(len(chars))

#Creates a dictionary that maps each unique character to its corresponding index.
plato={ch:i for i, ch in enumerate(chars)}
#Creates a dictionary that maps each unique index to its corresponding character.
socrates={i:ch for i, ch in enumerate(chars)}

#'s' is the input string it takes in, 'for c in s' loops over each character in the input string, and `plato[c]` converts it to its mapped number.
encode = lambda s: [plato[c] for c in s]
decode = lambda l: ''.join(socrates[n] for n in l)

#Converts it into a tensor array (required for most tensor operations).
data_array = torch.tensor(encode(text), dtype=torch.long)

#Finds the index of value in the array at 90th percent.
cutoff = (int(0.9*len(data_array)))

#Limits the variable 'train_data' to only contain 90% of the total data_array.
train_data = data_array[:cutoff]
#Assigns the leftover of data_array as validation data.
val_data = data_array[cutoff:]

def get_batch(split):
    #Decides if 'data' is 'train_data' or 'val_data' based on the arguments passed through 'get_batch()'.
    data = train_data if split == 'train' else val_data

    # Picks 32 random starting positions (because batch_size = 32) from the training data.
    # Each value in randpoints is an index (a position in the data), NOT a value from the data.
    randpoints = torch.randint(len(data) - block_size, (batch_size,))

    # For each random starting index, grab a sequence of 16 characters (not the index itself, but the values from those positions).
    # Example: if i = 2, this grabs data[2:18] → 16 character *values* starting at position 2.
    x = torch.stack([data[i:i + block_size] for i in randpoints])

    # y is just like x, but shifted one character to the right.
    # It represents the "target" next characters for each input in x.
    y = torch.stack([data[i + 1: i + block_size + 1] for i in randpoints])
    
    # Note: the three above lines are what randomly pick chunks from either train_data or val_data to to train the AI on.

    x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def estimate_loss():

    # Dictionary to store the average losses of train data and val data.
    out = {}
    
    # Set the model to evaulation mode for accurate results.
    m.eval()

    # Loops twice, once using train data, once using val data.
    for split in ['train', 'val']:

        # Creates a one dimensional empty tensor filled with zeros, with a length of eval_iters.
        losses = torch.zeros(eval_iters)

        # Loop as many times as the value of eval_iters.
        for k in range(eval_iters):

            # Retrieves either train data or val data depending on which loop it is in (check line 96).
            # Assigns the output returned from get_batch to X, Y.
            # The out put is the 2D tensor of batch_size x block_size. 
            # Y is the X offsetted by 1 index forward, so Y is the target for X of the same index.
            X, Y = get_batch(split)

            # Passes the result of get_batch to the forward function in the bigram model.
            # It returns a 2D tensor of the data (either train or val) and it's loss compared to the correct target token.
            logits, loss = m(X, Y)

            # Converts the loss tensor into a normal python number.
            losses[k] = loss.item()
        
        # Returns the average of all the losses in the loop, what every eval_iters is set to, and saves it to the 'out' dictionary.
        # Either saves the average loss of train data or val data depending on what loop it's in (check line 96).
        out[split] = losses.mean()
    
    # Return the model to training mode so no errors occur.
    m.train()
    
    # Return the dictionary containing the average losses of both train and val data.
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Assigns each dimensional of 3D tensor (containing both token & postional embedding) to three seperate variables.
        B, T, C = x.shape

        # Creates a new tensor with the same shape as the input tensor.
        # Serves the purpose of representing what it contains so it can be queried.
        k = self.key(x)

        # Creates a new tensor with the same shape as the input tensor.
        # Serves the purpor of representing what each token queries.
        q = self.query(x)

        # Compare each query with all keys to find the best match for each. Providing each a score.
        # For each of the 32 sequences in the batch, each of the 16 tokens in that sequence, you get 16 raw scores.
        # They represent how much that token "likes" each other token (including itself).
        # Then multiplies each score by a decimal value to make smaller and easier to softmax.
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # Prevents each token from getting context from future tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Turns the raw attention scores into probabilities.
        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        # Assigns a list of 32 unique values for each token in case it is chosen.
        v = self.value(x)

        # Applies it the above line to wei.
        out = wei @ v
        
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Creates 4 attention heads.
        # Each head takes in (32, 64, 128) and returns (32, 64, 32).
        # The 4 outputs are then joined together into (32, 64, 128).
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Takes the joined (32, 64, 128) from all heads and mixes their info together.
        # The shape stays (32, 64, 128), but now each position sees across all heads.
        self.proj = nn.Linear(n_embd, n_embd)

        # Randomly removes values from the (32, 64, 128) tensor to prevent overfitting.
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Passes input through all 4 heads, each returning (32, 64, 32)
        # Joins them side-by-side into (32, 64, 128)
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Calls 'self.proj' to mix the 4 heads' results together while keeping shape (32, 64, 128) while randomly dropping some values during.
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()

        # First makes the last number in the tensor bigger (128 → 512) so it can more accurate tweak the probabilites.
        # Then brings it back to 128 to match the original shape.
        # Final shape stays the same, but with more accurate values.
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        # Passes the input through 'self.net'.
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        # Splits the 128 features evenly across 4 heads (32 features per head.)
        # Helps seperate head focus on seperate parts of the each token's feature and learn more specific patterns.
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Wrap the entire model in a class so that it can contain all of the model settings (it is best practice for pytorch).
class BigramLanguageModel(nn.Module):

    #__init__ can be thought of as the ingredients/settings of the model. 
    # Invalid -> (check below fix) Keep in mind all use vocab_size inside the model is not directly calling 'vocab_size' defined at line 11 but serves as a arbritary variable name to store the argument passed in 'model = BigramLanguageModel(vocab_size)'
    # since 'vocab_size' is already defined globally you don't need to pass it as an argument.
    def __init__(self):

        #Needed to tell pytorch not to overwrite its own settings within the model's setting but to initialize its own setting and then run the model's setting.
        super().__init__()   
        
        # Creates a 65x128 tensor representing all unique characters, each with 32 unique features.
        self.embedding_table = nn.Embedding(vocab_size, n_embd)

        # Creates a 2D tensor (16x128) to represent a unique 128 features for each possible position (0-17).
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        # Compares the 128 unique features of each character with each of the 65 possible character to provide 65 scores
        # Each score represents the likelyhood of that character being the next token.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    # Note: the value of idx depends on where is this function is being called and what is being passed. 
    # You can call this function multiple times, with each output being totally unrelated to the other.
    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Takes 't_batch', a 32x16 tensor (32 batches, 16 tokens each), stored in the variable idx.
        # The tensor is then passed through the embedding table to attach 32 unique features to each token.
        tok_emb = self.embedding_table(idx)

        # Assigns the 2D tensor (32x16) representing each unique position to 'pos_emb'.
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # Meshes the token embedding and the position embedding to create a unique tensor for each token at each position.
        # Even if its the same token but with different position, it will be represented by a different tensor.
        x = tok_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)

        # Calls 'self.lm_head' to assign 65 scores to each token corresponding to each possible next token after comparing it with the 128 unique feature of the tokens in the tensor containing the training data. 
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            # Assigns each of the three dimension of the tensor as B (batch size), T (block size), C(all possibly probabilities).
            B, T, C = logits.shape

            # Flattens a 32 batch, 16 values each, 65 probabilities per value tensor into a 1 batch, 512 values total, 65 probabilities per value tensor.
            logits = logits.view(B*T, C)

            # Flattens a 32 batch, 16 values each tensor to a  1 batch, 512 values total 2D tensor.
            targets = targets.view(B*T)

            # From all the possibile probabilities after softmaxxing, it extracts the model's confidence on the correct next token to compute the loss.
            # Note: The way in which the model converts back the probability to the character is because it's stored like the following:
            # token = [0.7, 0.1, 0.2] so when the model chooses 0.7 it means the next token is likely at index 0. 
            # It then takes from the list of 65 characters, the character at index 0.
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        # Loops as many times as max_new_tokens is set to, each loop using the token predicted in the last loop to predict the next.
        for _ in range(max_new_tokens):

            # Crop input to the last block_size tokens to avoid positional embedding overflow.
            idx_cond = idx[:, -block_size:]

            # Pass the cropped input to the model's forward method (triggered by self(idx_cond)),
            # It returns the logits (a three dimensional tensor consisting of batch, tokens in the batch, and 65 raw probabilities per token) and loss of idx which is then assigned to to variable of the same name respectively.
            logits, loss = self(idx_cond)

            #It takes only takes the last token from all the batches alongside their 65 raw probability distribution.
            # : means the same as 0: its just cleaner.
            # adding : is needed to correctly place -1 in the middle which is where it needs to be to reference the sequence and not the batch or raw probability distribution.
            logits = logits[:, -1, :]
            
            # converts the logits (3 dimensional) into probabilities totalling 1 (100%). 
            # dim=-1 sets it to the last dimension so you don't have to manually count of many dimensions there are. 
            # The last dimension is the raw probability distribution. 
            probs = F.softmax(logits, dim=-1)

            # Randomly picks a number from the probabilty distrubtion, thus selecting the next token.
            # It is random but weighted by the probabilities. 0.6 has a higher chance to get pick than 0.2 but 0.2 might still get picked.
            # This is what allows GPT models to give different answers to the same question. Essentially, it creates creativity.
            idx_next = torch.multinomial(probs, num_samples=1)

            # Adds the new predicted token to idx within the dimension tokens are stored (one down from batch).
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    
#This passes the global 'vocab_size's value to the model's local 'vocab_size.'
#Note: 'def __init' in the model is triggered here once automatically to setup the model. Any subsequent call to the model won't automatically trigger it again. It's a feature of python.
model = BigramLanguageModel()
m = model.to(device)

# Uses the AdamW optimizer. This is the part where the model uses gradient descent to learn.
# model.parameters() passes to the model which weights needed to be updated for lower loss.
# A learning rate of 1e-3 0.001. So if the gradient (proposed change to weights by AdamW is -0.02, you times it with the learning rate).
# This allows for smaller steps within gradient space so the model doesnt overshoot and fall into a local minima's.
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(max_steps):

    # This checks if it’s time to evaluate the model’s performance.
    # eval_interval is how often you want to evaluate (e.g. every 300 steps)
    if steps % eval_iters == 0:

        # Calls the estimate_loss func which returns the dictionary containing the average loss for both train and val data. 
        # Assigns it to the variable 'losses'.
        losses = estimate_loss()

        # Prints losses alongside the current step.
        print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Assigns the first output of get_batch (which is x) and the second output of get_batch (which is y) to the corresponding value of c_batch & t_batch. 
    # Meaning, c_batch = x & t_batch = y.
    t_batch, v_batch = get_batch('train')
    
    # Passes both the training data and the validation data into the model and expects two outputs: 
    # 1. A transformed logits (the flattened embedding table)
    # 2. The loss between the predicted next token and the actual target token.
    logits, loss = m(t_batch, v_batch)

    # Clears old gradients from the last step.
    # Gradients are how much the optimizer tells the embedding table to tweak weights by. Stored in model.embedding_table.weight.grad
    optimizer.zero_grad(set_to_none=True)

    # Calculates the gradients: how much each weight contributes to the current loss.
    loss.backward()

    # This applies the gradients: it actually updates the model’s weights.
    optimizer.step()
    
    # Takes the average loss outputted by the forward function in the model (a 1-element tensor) and converts it into a plain python number.
    # print(loss.item())

# Save the model.
torch.save(m.state_dict(), 'shakespeare_gpt.pth')

# Creates the prompt: a tensor of shape (1, 1) containing just the token [[0]]. torch.zeros automatically fills an empty tensor with 0's.
# dtype=torch.long ensures the input is an integer and not a float.
# device=device moves it to CPU or GPU depending on if the GPU exists/compatible.
context = torch.zeros((1, 1), dtype=torch.long, device=device)

# max_new_tokens determines the amount of loops in the generate function called within the model, aka how many predictions it'll make.
# [0] selects the first (and only) token in that input row. If you added an additional input row with its own token, you would need to do [0] and [1] and so on the more rows you add.
# .tolist() converts the tensor to a regular Python list of token indices.
# decode converts that list of token indices back into readable characters using the vocab map and prints it.
# Viva la GPT. Holy shit that took forever.
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
