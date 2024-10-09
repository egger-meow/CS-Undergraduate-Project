import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from pathlib import Path

from settings import channels, startChannel, timeStamps, batchSize

class Vibration(object):
    # read raw Vibration file with 6 channel (not used now)
    def __init__(self, path = 'E:/download/20240604/20240604/vibdata/20240604235739.csv', trainDataRatio = 0.8):
        df = pd.read_csv(path)
        desiredSampleRate = 300
        originalSampleRate = 8192
        totalSeconds = 30
        downSampleFactor = originalSampleRate // desiredSampleRate
        
        dataRawNumpy = df.iloc[::downSampleFactor, 2:8].head(desiredSampleRate * totalSeconds).values
        dataNormalizedNumpy = self.normalization(dataRawNumpy)
        
        sampleLength = timeStamps
        stride = 20
        dataSlidingWindow = self.slidingWindow(dataNormalizedNumpy ,sampleLength, stride).transpose(0,2,1)

        dataTensor = torch.tensor(dataSlidingWindow, dtype = torch.float32)  
        
        numSamples = dataTensor.shape[0]

        trainSize = int(trainDataRatio * numSamples) if int(trainDataRatio * numSamples) > 0 else 1

        trainData, testData = dataTensor[:trainSize], dataTensor[trainSize:]
        print(trainData.shape)
        
        batchSize = 32
        self.train_loader = DataLoader(TensorDataset(trainData), batch_size=batchSize, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(testData), batch_size=batchSize, shuffle=False)
        
    
    def __init__(self, dir = 'D:/leveling/leveling_data/Normal/Amp/state345/', trainDataRatio = 0.85, displayData = False, test = False):
        # read files from a dir, each file represent a sample
        directory = Path(dir)

        files = [f for f in directory.iterdir() if f.is_file() and f.name != '.DS_Store' ]
        dataList = []
        
        for file in files:
            path = dir + file.name 
            df = pd.read_csv(path)
            data = df.iloc[:, startChannel: startChannel + channels].values.transpose(1,0)

            dataToList = list(data)
            for d in range(startChannel, startChannel + channels):
                # print(d)
                dimData = dataToList[d - startChannel]
                
                ampChannel = 0
                
                if d == ampChannel:
                    flatStart = self.getFlatStart(dimData)
                    if flatStart is not None:
                        fdata = dimData[:flatStart]
                        # dimData = dimData[:flatStart] 
                if displayData:
                    plt.figure(figsize=(10, 6))
                    plt.plot(dimData, label=f'Original Data (channel-{d})')
                    if d == ampChannel and  flatStart is not None:
                        plt.plot(range(flatStart), fdata, label='Filtered Wavy Part', color='r')
                    plt.title(f'channel-{d}')
                    plt.show()
                
                adjusted_lens = timeStamps
                dimData = self.interpolation(adjusted_lens, dimData)
                min_value = np.min(dimData)
                max_value = np.max(dimData)

                dimData = (dimData - min_value) / (max_value - min_value)
                
                dataToList[d - startChannel] = dimData
                
            data = np.array(dataToList)
            dataList.append(data) 

        dataTensor = torch.tensor(
            np.array(dataList), 
            dtype = torch.float32)  
        # not normalization yet
        
        # actually test data here is validation data
        numSamples = dataTensor.shape[0]

        trainSize = int(trainDataRatio * numSamples) if not test else 0

        trainData, testData = dataTensor[:trainSize], dataTensor[trainSize:]
        
        bs = batchSize if not test else 1
        
        if not test:
            self.train_loader = DataLoader(TensorDataset(trainData), batch_size=bs, shuffle=False)
        self.test_loader = DataLoader(TensorDataset(testData), batch_size=bs, shuffle=False)
        
        print(trainData.shape, testData.shape)
        
    def slidingWindow(self, data, sampleLength, stride):
        numWindows = (data.shape[0] - sampleLength) // stride + 1 
        windows = np.array([data[i:i + sampleLength] for i in range(0, numWindows * stride, stride)])
        return windows 
    
    def interpolation(self, newDim, data):
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, newDim)

        return np.interp(x_new, x_old, data)
    
    def normalization(self, data, channels = 6):
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        return normalized
    
    def getFlatStart(self, data):
        flat_threshold = 0.01
        window_size = 5
        # print(diff)
        for i in range(0, data.shape[0], window_size):
            if np.std(data[i: i+window_size]) < flat_threshold:
                return i+window_size*2
        return None
# test = Vibration(dir = 'D:/leveling/leveling_data/v1/Normal/stage345/', displayData=True)


