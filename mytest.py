import torch
from networks import Generator
from configs import Config

conf = Config().parse()
model = Generator(conf)
inp=torch.randn((1, 3, 128 , 128 ,4))
op=model(inp)
print("Op shape: ",op.shape)