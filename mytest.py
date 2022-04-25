from configs import Config
from data import DataGenerator
from kernelGAN import KernelGAN
from learner import Learner
import tqdm

def test(conf):
    gan = KernelGAN(conf)
    learner = Learner()
    data = DataGenerator(conf, gan)
    for iteration in tqdm.tqdm(range(conf.max_iters), ncols=60):
        [g_in, d_in] = data.__getitem__(iteration)
        gan.train(g_in, d_in)
        learner.update(iteration, gan)
        break
conf=Config().parse()
test(conf)
