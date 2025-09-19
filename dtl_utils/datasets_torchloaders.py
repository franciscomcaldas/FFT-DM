import torch
from torch.utils.data import Dataset
import numpy as np
import os




class ETTm1TrainingDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/ETTm1/train_ettm1_1056.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        #training_data = np.split(training_data, 571, 0)
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
    
class SolarEnergyTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/Solar_Energy/y_train.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)
    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class SolarEnergyValidationDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/Solar_Energy/y_val.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['val_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)
        

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class SolarEnergyTestDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/Solar_Energy/y_test.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['test_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class TempRainTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/TempRain/y_train.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx,1:]
    
class TempRainValidationDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/TempRain/y_val.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['val_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx,1:]

class TempRainTestDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/TempRain/y_test.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['test_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx,1:]
    
class Synth1TrainingDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/Synth1/y_train.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)
        

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
                
class Synth1ValidationDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/Synth1/y_val.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['val_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)
        

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]

class Synth1TestDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/Synth1/y_test.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['test_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class SolarEnergyTestDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/solar-energy/y_test.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['test_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class SolarEnergyTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/solar-energy/y_train.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class SolarEnergyValidationDataset(Dataset):
    def __init__(self, trainset_config):
        ettm1_path = "/data/<user>/datasets/solar-energy/y_train.npy"
        if os.path.exists(ettm1_path):
            print('Importing from data server...')
            training_data = np.load(ettm1_path)
        else:
            training_data = np.load(trainset_config['val_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().unsqueeze(-1)

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]


class ElectricityTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        dat_path = "/data/<user>/datasets/Electricity/train_electricity.npy"
        if os.path.exists(dat_path):
            print('Importing from data server...')
            training_data = np.load(dat_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        option1_data = training_data.reshape(-1, 100,37)
        option2_data = option1_data.transpose(0,2,1).reshape(-1,1, 100)
        option2_data = option2_data.transpose(0,2,1)
        #option2_data = np.split(option2_data, 370, 0)
        option2_data = np.array(option2_data)
        self.training_data = torch.from_numpy(option2_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
                

class PTBXLTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        dat_path = "/data/<user>/datasets/PTB-XL-1000/train_ptbxl_1000.npy"
        if os.path.exists(dat_path):
            print('Importing from data server...')
            training_data = np.load(dat_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        # Transpose to match the desired shape
        training_data = training_data.transpose(0,2,1)
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float()

    def __len__(self):
        return self.training_data.size(0)

    def __getitem__(self, idx):
        return self.training_data[idx]
    
class MujocoTrainingDataset(Dataset):
    def __init__(self, trainset_config):
        dat_path = "/data/<user>/datasets/Mujoco/train_mujoco.npy"
        if os.path.exists(dat_path):
            print('Importing from data server...')
            training_data = np.load(dat_path)
        else:
            training_data = np.load(trainset_config['train_data_path'])
        training_data = np.array(training_data)
        self.training_data = torch.from_numpy(training_data).float().cuda()

    def __len__(self):
        return self.training_data.size(0)
    
    def __getitem__(self, idx):
        return self.training_data[idx]
    
class MujocoTestDataset(Dataset):
    def __init__(self, testset_config):
        dat_path = "/data/<user>/datasets/Mujoco/test_mujoco.npy"
        if os.path.exists(dat_path):
            print('Importing test data from data server...')
            test_data = np.load(dat_path)
        else:
            test_data = np.load(testset_config['test_data_path'])
        test_data = np.array(test_data)
        self.test_data = torch.from_numpy(test_data).float().cuda()
    def __len__(self):
        return self.test_data.size(0)
    
    def __getitem__(self, idx):
        return self.test_data[idx]

class ETTm1TestDataset(Dataset):
    def __init__(self, testset_config):
        ettm1_test_path = "/data/<user>/datasets/ETTm1/test_ettm1_1056.npy"
        if os.path.exists(ettm1_test_path):
            print('Importing test data from data server...',flush=True)
            test_data = np.load(ettm1_test_path)
        else:
            test_data = np.load(testset_config['test_data_path'])
        test_data = np.array(test_data)
        self.test_data = torch.from_numpy(test_data).float()

    def __len__(self):
        return self.test_data.size(0)

    def __getitem__(self, idx):
        return self.test_data[idx]


class ElectricityTestDataset(Dataset):
    def __init__(self, testset_config):
        dat_path = "/data/<user>/datasets/Electricity/test_electricity.npy"
        if os.path.exists(dat_path):
            print('Importing test data from data server...')
            test_data = np.load(dat_path)
        else:
            test_data = np.load(testset_config['test_data_path'])
        option1_data = test_data.reshape(-1, 100, 37)
        option2_data = option1_data.transpose(0, 2, 1).reshape(-1, 1, 100)
        option2_data = option2_data.transpose(0, 2, 1)
        option2_data = np.array(option2_data)
        self.test_data = torch.from_numpy(option2_data).float()

    def __len__(self):
        return self.test_data.size(0)

    def __getitem__(self, idx):
        return self.test_data[idx]


class PTBXLTestDataset(Dataset):
    def __init__(self, testset_config):
        dat_path = "/data/<user>/datasets/PTB-XL-1000/test_ptbxl_1000.npy"
        if os.path.exists(dat_path):
            print('Importing test data from data server...')
            test_data = np.load(dat_path)
        else:
            test_data = np.load(testset_config['test_data_path'])
        # Transpose to match the desired shape
        test_data = test_data.transpose(0, 2, 1)
        test_data = np.array(test_data)
        self.test_data = torch.from_numpy(test_data).float()

    def __len__(self):
        return self.test_data.size(0)

    def __getitem__(self, idx):
        return self.test_data[idx]



                
