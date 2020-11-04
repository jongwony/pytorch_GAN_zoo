import os

from models.progressive_gan import ProgressiveGAN
from models.trainer.progressive_gan_trainer import ProgressiveGANTrainer


if __name__ == '__main__':
    pathdb = os.path.join('output')
    trainer = ProgressiveGANTrainer(pathdb=pathdb, useGPU=True, checkPointDir='output_networks')
    # trainer.model = ProgressiveGAN(useGPU=True, storeAVG=True)
    # trainer.model.load_state_dict(torch.load(checkPointDir))
    trainer.train()

