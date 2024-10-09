import torch
# ---- data path definitions ----

norm_trainDataDir = 'D:/leveling/leveling_data/v1/Normal/train/'
abnorm_trainDataDir = 'D:/leveling/leveling_data/v1/Abnormal/train/'

norm_testDataDir = 'D:/leveling/leveling_data/v1/Normal/test/'
abnorm_testDataDir = 'D:/leveling/leveling_data/v1/Abnormal/test/'

autoencoderNormPath = 'D:/leveling/pytorch-AE/checkpoints/autoEncoderNorm.pth'
autoencoderAbnormPath = 'D:/leveling/pytorch-AE/checkpoints/autoEncoderAbnorm.pth'

cuda = torch.cuda.is_available()

# ---- training selection ----

trainNormalAutoEncoder = True # choose to train normal or abnormal autoencoder

# ---- hyper parameters ----

channels = 7
startChannel = 0
timeStamps = 100

epochs = 100
batchSize = 16

lr = 0.001
scheduler_stepSize = 10
scheduler_gamma = 0.85