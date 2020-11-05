import os

import torch
import torchvision.utils as vutils

from models.progressive_gan import ProgressiveGAN


path = os.path.join('output_networks', 'GAN_s3_i96000.pt')

model = ProgressiveGAN(useGPU=True, storeAVG=True)

model.load_state_dict(torch.load(path))

noiseData, _ = model.buildNoiseData(1)

fake_im = model.test(noiseData, getAvG=True, toCPU=False).detach()

vutils.save_image(fake_im, 'test3.png', normalize=True)