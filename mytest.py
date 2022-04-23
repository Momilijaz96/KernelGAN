from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
import tqdm

def test(conf):
   
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        print("Generator input: ",g_in.shape)
        print("Discriminator input: ",d_in.shape)
        break
conf=Config().parse()
test(conf)
