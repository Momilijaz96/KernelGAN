import torch
from networks import Generator
from configs import Config

conf = Config().parse()
model = Generator(conf)
inp=torch.randn((1, 12, 128 , 128)) #b x 3*t x h x w
op=model(inp)
print("Op shape: ",op.shape)