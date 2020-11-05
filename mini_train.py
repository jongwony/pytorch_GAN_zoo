import os

from models.trainer.progressive_gan_trainer import ProgressiveGANTrainer
from models.utils.utils import getLastCheckPoint



if __name__ == '__main__':
    checkPointDir = os.path.join('output_networks')
    pathdb = os.path.join('output')
    trainer = ProgressiveGANTrainer(pathdb=pathdb, useGPU=True, saveIter=16000, lossIterEvaluation=100, checkPointDir=checkPointDir)
    # If a checkpoint is found, load it
    trainConfig, pathModel, pathTmpData = getLastCheckPoint(checkPointDir, 'GAN')
    trainer.loadSavedTraining(pathModel, trainConfig, pathTmpData)
    trainer.train()

