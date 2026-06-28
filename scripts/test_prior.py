import time
import torch
from prior import rand_dataset_filtered, rand_cat_sizes

start = time.time()
x_cat_sizes = rand_cat_sizes(10)
print(f"x_cat_sizes: {x_cat_sizes}")
n_classes = 4
y_cat_sizes = [n_classes]
n_samples = 1000

for i in range(4):
    tensors = rand_dataset_filtered(x_cat_sizes, y_cat_sizes, n_samples)
    x = torch.cat([tensors[f"x_{j}"] for j in range(len(x_cat_sizes))], dim=-1)
    y = tensors["y_0"].squeeze(-1)
    print(f"Dataset {i}: x.shape={x.shape}, y.shape={y.shape}")

print(f"Time for 4 datasets: {time.time() - start:.2f}s")
