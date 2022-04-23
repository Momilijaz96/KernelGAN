import torch
from networks import Generator

model = Generator()
inp=torch.randn((1, 3, 128 , 128 ,4))
op=model(inp)
print("Op shape: ",op.shape)