import os
import torch
import numpy as np
from PIL import Image
import jsonlines
import random
from torch.utils.data import Dataset

image_column = "file_name"

class ThyroidDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom"
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {"TR2-3": 0, "TR4": 1, "TR5": 2}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition: 
                    if self.image_encode == "Custom":
                        d["ref_im_file"] = os.path.join(data_dir, d[image_column])
                    elif self.image_encode == "MedSAM":
                        d["ref_im_emb"] = os.path.join(data_dir.replace("TUSC_data", "TUSC_data_emb"), d[image_column].replace("png", "npy")) # our method
                
                d[image_column] = os.path.join(data_dir, d[image_column])
                d["cls"] = self.grading_to_id[d["cls"]]
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def keep_and_drop(self, data, keep_all_prob, drop_all_prob, drop_each_prob):
        """
        Returns:
            Dict: data & its condition (path)
        """
        seed = random.random()
        conditions = ["ref_im_emb"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            results = data
        else:
            for i in range(len(conditions)):
                if random.random() < drop_each_prob[i]:
                    data[conditions[i]] = 1
            results = data
        return results

    def __getitem__(self, index):
        return self.keep_and_drop(self.data_list[index], self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        # return self.data_list[index]

class ACDCDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom"
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition: 
                    if self.image_encode == "Custom":
                        d["ref_im_file"] = os.path.join(data_dir, d[image_column]) 
                    elif self.image_encode == "MedSAM":
                        d["ref_im_emb"] = os.path.join(data_dir.replace("ACDC_data", "ACDC_data_emb"), d[image_column].replace("png", "npy")) # our method
                
                d[image_column] = os.path.join(data_dir, d[image_column])
                d["cls"] = self.grading_to_id[d["cls"]]
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def keep_and_drop(self, data, keep_all_prob, drop_all_prob, drop_each_prob):
        """
        Returns:
            Dict: data & its condition (path)
        """
        seed = random.random()
        conditions = ["ref_im_emb"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            results = data
        else:
            for i in range(len(conditions)):
                if random.random() < drop_each_prob[i]:
                    data[conditions[i]] = 1
            results = data
        return results

    def __getitem__(self, index):
        return self.keep_and_drop(self.data_list[index], self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        # return self.data_list[index]
        
class MosmedDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom"
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {"CT-0": 0, "CT-1": 1, "CT-234": 2}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition: 
                    if self.image_encode == "Custom":
                        d["ref_im_file"] = os.path.join(data_dir, d[image_column])
                    elif self.image_encode == "MedSAM":
                        d["ref_im_emb"] = os.path.join(data_dir.replace("MosMed_data", "MosMed_data_emb"), d[image_column].replace("png", "npy")) # our method
                
                d[image_column] = os.path.join(data_dir, d[image_column])
                d["cls"] = self.grading_to_id[d["cls"]]
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def keep_and_drop(self, data, keep_all_prob, drop_all_prob, drop_each_prob):
        """
        Returns:
            Dict: data & its condition (path)
        """
        seed = random.random()
        conditions = ["ref_im_emb"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            results = data
        else:
            for i in range(len(conditions)):
                if random.random() < drop_each_prob[i]:
                    data[conditions[i]] = 1
            results = data
        return results

    def __getitem__(self, index):
        return self.keep_and_drop(self.data_list[index], self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        # return self.data_list[index]
    
class MRNetDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom"
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {0: 0, 1: 1}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition: 
                    if self.image_encode == "Custom":
                        d["ref_im_file"] = os.path.join(data_dir, d[image_column]) 
                    elif self.image_encode == "MedSAM":
                        d["ref_im_emb"] = os.path.join(data_dir.replace("MRNet_data", "MRNet_data_emb"), d[image_column].replace("png", "npy")) # our method
                
                d[image_column] = os.path.join(data_dir, d[image_column])
                d["cls"] = self.grading_to_id[d["cls"]]
                data_list.append(d)
        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def keep_and_drop(self, data, keep_all_prob, drop_all_prob, drop_each_prob):
        """
        Returns:
            Dict: data & its condition (path)
        """
        seed = random.random()
        conditions = ["ref_im_emb"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            results = data
        else:
            for i in range(len(conditions)):
                if random.random() < drop_each_prob[i]:
                    data[conditions[i]] = 1
            results = data
        return results

    def __getitem__(self, index):
        return self.keep_and_drop(self.data_list[index], self.keep_all_cond_prob, self.drop_all_cond_prob, self.drop_each_cond_prob)
        # return self.data_list[index]
    
    
def load_image(image_path):
    ref_image = Image.open(image_path).convert("L")
    return ref_image