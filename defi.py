# ---- data definitions ----

channels = 6
startChannel = 1
timeStamps = 100


norm_trainDataDir = 'D:/leveling/leveling_data/v1/Normal/train/'
abnorm_trainDataDir = 'D:/leveling/leveling_data/v1/Abnormal/train/'

norm_testDataDir = 'D:/leveling/leveling_data/v1/Normal/test/'
abnorm_testDataDir = 'D:/leveling/leveling_data/v1/Abnormal/test/'

autoencoderNormPath = 'D:/leveling/pytorch-AE/checkpoints/autoEncoderNorm.pth'
autoencoderAbnormPath = 'D:/leveling/pytorch-AE/checkpoints/autoEncoderAbnorm.pth'

# ---- hyper parameters ----

batchSize = 32

lr = 0.001
scheduler_stepSize = 15
scheduler_gamma = 0.85