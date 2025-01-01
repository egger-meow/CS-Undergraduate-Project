import torch
from datetime import datetime

# ---- training selection ----

autoEncoder = 'AE'
architechture = 'CNN1D'

# ---- data path definitions ----

dataVerion = 'v3' # v0, v1, v2, v3

norm_sourceDir = f'D:/leveling/leveling_data/{dataVerion}/source/Normal'
abnorm_sourceDir = f'D:/leveling/leveling_data/{dataVerion}/source/Abnormal'

norm_trainDataDir = f'D:/leveling/leveling_data/{dataVerion}/Normal/train/'
abnorm_trainDataDir = f'D:/leveling/leveling_data/{dataVerion}/Abnormal/train/'

norm_testDataDir = f'D:/leveling/leveling_data/{dataVerion}/Normal/test/'
abnorm_testDataDir = f'D:/leveling/leveling_data/{dataVerion}/Abnormal/test/'

current_date = datetime.now().strftime("%y%m%d%H")
autoencoderNormPath = f'D:/leveling/pytorch-AE/checkpoints/aeNorm_{dataVerion}_{architechture}_{current_date}.pth'
autoencoderAbnormPath = f'D:/leveling/pytorch-AE/checkpoints/aeAbnorm_{dataVerion}_{architechture}_{current_date}.pth'

autoencoderNormPath = f'D:/leveling/pytorch-AE/checkpoints/aeNorm_{dataVerion}_{architechture}_25010104.pth'
autoencoderAbnormPath = f'D:/leveling/pytorch-AE/checkpoints/aeAbnorm_{dataVerion}_{architechture}_25010104.pth'

cuda = torch.cuda.is_available()

testFileNum = 50

testingShapeBias = False
# testingShapeBias = True

# ---- data preparing ----

sampleRate = 256
sampleRate_origin = 8192
fft = False

slidingWindow_aeNorm = False    # if true, the window size will be timeStamps, 
slidingWindow_aeAbnorm = False   # if true, the window size will be timeStamps, 
stride = 128             # looping through the data with sampleRate

# 0,    1,      2,      3,      4,      5,    6
# amp,  door-x, door-y, door-z, car-x, car-y, car-z
channelSelected = [1,2,3]             
channels = len(channelSelected) 

timeStamps = 1024

# ---- hyper parameters ----
epochs = 300
batchSize_aeNorm = 32
batchSize_aeAbnorm = 4
embeddingSize = 16

lr = 0.005
scheduler_stepSize = 8
scheduler_gamma = 0.85

vaeBias = 8


# ---- CNN1D parameters ----

decoderShapeBias = timeStamps - 8

# ---- LSTM parameters ----

dropout = 0.01
layers  = 1