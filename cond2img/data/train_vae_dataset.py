import os
import jsonlines
from torch.utils.data import Dataset

image_column = "file_name"
class ThyroidDataset(Dataset):
    def __init__(
            self,
            data_json_path: str,
            width: int = 256,
            height: int = 256
    ):
        self.width = width
        self.height = height
        self.data_list = self.load_jsonl(data_json_path)
    
    def load_jsonl(self, data_json_path):
        data_list = []
        with jsonlines.open(data_json_path,'r') as freaders:
            for d in freaders:
                d[image_column] = os.path.join("TUSC/TUSC_data", d[image_column])
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]

class ACDCDataset(Dataset):
    def __init__(
            self,
            data_json_path: str,
            width: int = 256,
            height: int = 256
    ):
        self.width = width
        self.height = height
        self.data_list = self.load_jsonl(data_json_path)
    
    def load_jsonl(self, data_json_path):
        data_list = []
        with jsonlines.open(data_json_path,'r') as freaders:
            for d in freaders:
                d[image_column] = os.path.join("ACDC/ACDC_data", d[image_column])
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    
class MosMedDataset(Dataset):
    def __init__(
            self,
            data_json_path: str,
            width: int = 256,
            height: int = 256
    ):
        self.width = width
        self.height = height
        self.data_list = self.load_jsonl(data_json_path)
    
    def load_jsonl(self, data_json_path):
        data_list = []
        with jsonlines.open(data_json_path,'r') as freaders:
            for d in freaders:
                d[image_column] = os.path.join("MosMedData/MosMed_data", d[image_column])
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
    
class MRNetDataset(Dataset):
    def __init__(
            self,
            data_json_path: str,
            width: int = 256,
            height: int = 256
    ):
        self.width = width
        self.height = height
        self.data_list = self.load_jsonl(data_json_path)
    
    def load_jsonl(self, data_json_path):
        data_list = []
        with jsonlines.open(data_json_path,'r') as freaders:
            for d in freaders:
                d[image_column] = os.path.join("MRNet/MRNet_data", d[image_column])
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        return self.data_list[index]
