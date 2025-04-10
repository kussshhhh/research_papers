import torch 
import numpy 

x = torch.tensor([[1,2], [2,3], [-1,1], [-2, 2]], dtype=torch.float32)
y = torch.tensor([1,1,0,0], dtype=torch.float32) 

print("X:", x)
print("y:", y)

w = torch.zeros(2, dtype=torch.float32)
b = torch.tensor(0., dtype=torch.float32)
print("w:", w)
print("b:", b)

eta = 0.1

def step(z):
    return 1. if z>=0 else 0.

for epoch in range(10):
    for i in range(len(x)):
        z = torch.dot(w, x[i]) + b
        good = step(z)

        if(good != y[i]):
            w += eta * (y[i] - good) * x[i]
            b += eta * (y[i] - good)

        print(f"epoch {epoch}, sample {i}, weights: {w}, bias: {b}")

# testing
for i in range(len(x)):
    z = torch.dot(w, x[i]) + b
    print(f"point {x[i]} predicted: {step(z)}, true: {y[i]}")