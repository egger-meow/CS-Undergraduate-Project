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

architechture = 'LSTM'

# ---- hyper parameters ----

channels = 3
startChannel = 1
timeStamps = 100

epochs = 50
batchSize = 16

embeddingSize = 128

lr = 0.005
scheduler_stepSize = 10
scheduler_gamma = 0.7

# ---- LSTM parameters ----

dropout = 0.0
layers  = 1