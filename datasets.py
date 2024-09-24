import torch
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset, random_split


class MNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

class EMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=True, download=True, split='byclass',
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('data/emnist', train=False, split='byclass',
            transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

class FashionMNIST(object):
    def __init__(self, args):
        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fmnist', train=False, transform=transforms.ToTensor()),
            batch_size=args.batch_size, shuffle=True, **kwargs)

class Vibration(object):
    def __init__(self):
        df = pd.read_csv('E:/download/20240604/20240604/vibdata/20240604235739.csv')
        desiredSampleRate = 300
        originalSampleRate = 8192
        totalSeconds = 30
        downSampleFactor = originalSampleRate // desiredSampleRate
        
        dataRawNumpy = df.iloc[::downSampleFactor, 2:8].head(desiredSampleRate * totalSeconds).values
        dataNormalizedNumpy = self.normalization(dataRawNumpy)
        
        sampleLength = 100
        stride = 10
        dataSlidingWindow = self.slidingWindow(dataNormalizedNumpy ,sampleLength, stride).transpose(0,2,1)

        dataTensor = torch.tensor(dataSlidingWindow, dtype = torch.float32)

        numSamples = dataTensor.shape[0]

        trainSize = int(0.8 * numSamples)

        trainData, testData = dataTensor[:trainSize], dataTensor[trainSize:]
        # print(trainData.shape)
        
        batchSize = 32
        self.train_loader = DataLoader(TensorDataset(trainData), batch_size=batchSize, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(testData), batch_size=batchSize)
        print('data prepared.')
        
    def slidingWindow(self, data, sampleLength, stride):
        numWindows = (data.shape[0] - sampleLength) // stride + 1 
        windows = np.array([data[i:i + sampleLength] for i in range(0, numWindows * stride, stride)])
        return windows 
    
    def normalization(self, data):
        dim0 = data.shape[0]
        # reshaped = data.reshape(-1, 6)
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(data)
        # print(data[np.argmax(normalized[:,2]),2])
        # print(normalized[0])
        return normalized.reshape(dim0, 6)
        

test = Vibration()
