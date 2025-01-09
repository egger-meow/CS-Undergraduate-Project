# settings.py
import os
import torch
from datetime import datetime

# ---- Training Selection ----
autoEncoder = os.getenv('AUTO_ENCODER', 'AE')
architecture = os.getenv('ARCHITECTURE', 'CNN1D')

# ---- Data Path Definitions ----
dataVersion = os.getenv('DATA_VERSION', 'v3')  # v0, v1, v2, v3

norm_sourceDir = f'D:/leveling/leveling_data/{dataVersion}/source/Normal'
abnorm_sourceDir = f'D:/leveling/leveling_data/{dataVersion}/source/Abnormal'

norm_trainDataDir = f'D:/leveling/leveling_data/{dataVersion}/Normal/train/'
abnorm_trainDataDir = f'D:/leveling/leveling_data/{dataVersion}/Abnormal/train/'

norm_testDataDir = f'D:/leveling/leveling_data/{dataVersion}/Normal/test/'
abnorm_testDataDir = f'D:/leveling/leveling_data/{dataVersion}/Abnormal/test/'

current_date = datetime.now().strftime("%y%m%d")
autoencoderNormPath = f'D:/leveling/pytorch-AE/checkpoints/aeNorm_{dataVersion}_{architecture}_{current_date}.pth'
autoencoderAbnormPath = f'D:/leveling/pytorch-AE/checkpoints/aeAbnorm_{dataVersion}_{architecture}_{current_date}.pth'

phaseII_trainSetPath = os.getenv('PHASEII_TRAIN_SET_PATH', 'D:/leveling/pytorch-AE/trainTest_II_dataSets/train_set_amp.joblib')
phaseII_testSetPath = os.getenv('PHASEII_TEST_SET_PATH', 'D:/leveling/pytorch-AE/trainTest_II_dataSets/test_set_amp.joblib')

phaseII_mode = os.getenv('PHASEII_MODE', 'amp')

# ---- General Settings ----
cuda = os.getenv('CUDA', 'True') == 'True' and torch.cuda.is_available()
testFileNum = int(os.getenv('TEST_FILE_NUM', 50))
testingShapeBias = os.getenv('TESTING_SHAPE_BIAS', 'False') == 'True'

# ---- Data Preparing ----
sampleRate = int(os.getenv('SAMPLE_RATE', 256))
sampleRate_origin = int(os.getenv('SAMPLE_RATE_ORIGIN', 8192))
fft = os.getenv('FFT', 'False') == 'True'

slidingWindow_aeNorm = os.getenv('SLIDING_WINDOW_AE_NORM', 'False') == 'True'
slidingWindow_aeAbnorm = os.getenv('SLIDING_WINDOW_AE_ABNORM', 'False') == 'True'
stride = int(os.getenv('STRIDE', 128))

channelSelected = list(map(int, os.getenv('CHANNEL_SELECTED', '0').split()))
channels = len(channelSelected)

timeStamps = int(os.getenv('TIME_STAMPS', 1024))

# ---- Hyper Parameters ----
epochs = int(os.getenv('EPOCHS', 300))
batchSize_aeNorm = int(os.getenv('BATCH_SIZE_AE_NORM', 32))
batchSize_aeAbnorm = int(os.getenv('BATCH_SIZE_AE_ABNORM', 4))
embeddingSize = int(os.getenv('EMBEDDING_SIZE', 8))

lr = float(os.getenv('LR', 0.005))
scheduler_stepSize = int(os.getenv('SCHEDULER_STEP_SIZE', 8))
scheduler_gamma = float(os.getenv('SCHEDULER_GAMMA', 0.85))

vaeBias = int(os.getenv('VAEBIAS', 8))

# ---- CNN1D Parameters ----
decoderShapeBias = timeStamps - 8

# ---- LSTM Parameters ----
dropout = float(os.getenv('DROPOUT', 0.01))
layers = int(os.getenv('LAYERS', 1))



channelSelected = [1,2,3]             
channels = len(channelSelected) 