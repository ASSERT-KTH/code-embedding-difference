import json
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset
from datasets import load_dataset

buggy_list = []
fixed_list = []

base_dir = "/mimer/NOBACKUP/groups/naiss2025-5-243/buggy_fixed_embeddings"

for chunk_num in range(19):
    file_path = f"{base_dir}/buggy_fixed_embeddings_chunk_{chunk_num:04d}.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        buggy_list.extend(data['buggy_embeddings'].tolist())
        fixed_list.extend(data['fixed_embeddings'].tolist())

import random

pairs = list(zip(buggy_list, fixed_list))
random.shuffle(pairs)

train_size = int(0.8 * len(pairs))
val_size = int(0.1 * len(pairs))

train_pairs = pairs[:train_size]
val_pairs = pairs[train_size:train_size+val_size]
test_pairs = pairs[train_size+val_size:]

train_buggy, train_fixed = zip(*train_pairs)
val_buggy, val_fixed = zip(*val_pairs)
test_buggy, test_fixed = zip(*test_pairs)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Dataset
class VectorPairDataset(Dataset):
    def __init__(self, buggy, fixed):
        self.buggy = torch.tensor(buggy, dtype=torch.float32)
        self.fixed = torch.tensor(fixed, dtype=torch.float32)
    def __len__(self):
        return len(self.buggy)
    def __getitem__(self, i):
        return self.buggy[i], self.fixed[i]

# Model
class MLP_Model(nn.Module):
    def __init__(self, input_size=1024, output_size=1024, hidden_sizes=[4096, 2048, 1024]):
        super().__init__()
        layers, in_f = [], input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_f, h), nn.ReLU()]
            in_f = h
        layers.append(nn.Linear(in_f, output_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# Cosine similarity loss
class CosineLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(dim=1)
    def forward(self, x, y):
        return 1 - self.cos(x, y).mean()
        
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loaders
train_loader = DataLoader(VectorPairDataset(train_buggy, train_fixed), batch_size=512, shuffle=True)
val_loader = DataLoader(VectorPairDataset(val_buggy, val_fixed), batch_size=512)
test_loader = DataLoader(VectorPairDataset(test_buggy, test_fixed), batch_size=512)

# Model
model = MLP_Model(input_size=train_buggy.shape[1], output_size=train_fixed.shape[1]).to(device)

# Loss + Optimizer
loss_fn = CosineLoss()
cos_fn = nn.CosineSimilarity(dim=1)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(20):
    model.train()
    tloss, tsim = 0, 0
    for b, f in train_loader:
        b, f = b.to(device), f.to(device)
        opt.zero_grad()
        out = model(b)
        loss = loss_fn(out, f)
        loss.backward()
        opt.step()
        tloss += loss.item() * b.size(0)
        tsim += cos_fn(out, f).mean().item() * b.size(0)
    tloss /= len(train_loader.dataset)
    tsim /= len(train_loader.dataset)

    model.eval()
    vloss, vsim = 0, 0
    with torch.no_grad():
        for b, f in val_loader:
            b, f = b.to(device), f.to(device)
            out = model(b)
            vloss += loss_fn(out, f).item() * b.size(0)
            vsim += cos_fn(out, f).mean().item() * b.size(0)
    vloss /= len(val_loader.dataset)
    vsim /= len(val_loader.dataset)

    print(f"{epoch+1}: train_loss={tloss:.6f}, val_loss={vloss:.6f}, "
          f"train_sim={tsim:.6f}, val_sim={vsim:.6f}")

# Final test
model.eval()
test_loss, test_sim = 0, 0
with torch.no_grad():
    for b, f in test_loader:
        b, f = b.to(device), f.to(device)
        out = model(b)
        test_loss += loss_fn(out, f).item() * b.size(0)
        test_sim += cos_fn(out, f).mean().item() * b.size(0)
print(f"Test: loss={test_loss/len(test_loader.dataset):.6f}, "
      f"sim={test_sim/len(test_loader.dataset):.6f}")

# Results

# 1: 0.008036 0.004522 0.991738 0.995386
# 2: 0.003950 0.003416 0.996012 0.996484
# 3: 0.003335 0.003217 0.996619 0.996733
# 4: 0.003058 0.003105 0.996890 0.996807
# 5: 0.002896 0.002867 0.997047 0.997053
# 6: 0.002789 0.002861 0.997152 0.997123
# 7: 0.002716 0.002706 0.997225 0.997248
# 8: 0.002664 0.002710 0.997275 0.997243
# 9: 0.002621 0.002554 0.997317 0.997424
# 10: 0.002597 0.002746 0.997341 0.997199
# 11: 0.002584 0.002653 0.997352 0.997274
# 12: 0.002570 0.002598 0.997367 0.997333
# 13: 0.002552 0.002513 0.997383 0.997428
# 14: 0.002546 0.002560 0.997389 0.997370
# 15: 0.002532 0.002610 0.997402 0.997327
# 16: 0.002526 0.002514 0.997407 0.997422
# 17: 0.002522 0.002632 0.997412 0.997296
# 18: 0.002517 0.002584 0.997417 0.997350
# 19: 0.002506 0.002595 0.997427 0.997332
# 20: 0.002501 0.002520 0.997432 0.997415
# Test: 0.002512 0.997420