import os
import torch
import numpy as np
import jsonlines
import cv2
import random
from torch.utils.data import Dataset
from einops import rearrange

video_column = "file_name"
        
class ThyroidVideoDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom",
            motion_condition: bool = False
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.motion_condition = motion_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {"TR2-3": 0, "TR4": 1, "TR5": 2}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition: 
                    initial_frm = d[video_column].replace("f", "").replace(".avi", ".npy")
                    d["ref_im_emb"] = os.path.join(data_dir.replace("TUSC_video", "TUSC_data_emb"), initial_frm)
                if self.motion_condition:
                    d["motion_file"] = os.path.join(data_dir.replace("TUSC_video", "TUSC_motion_field"), d[video_column].replace(".avi", ".npy")) 
                    
                d[video_column] = os.path.join(data_dir, d[video_column])
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
        conditions = ["ref_im_emb", "motion_file"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            data["motion_file"] = 1
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

class ACDCVolumeDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom",
            motion_condition: bool = False
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.motion_condition = motion_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition:
                    initial_frm = d[video_column].replace("_s", "_slice").replace(".avi", ".npy")
                    d["ref_im_emb"] = os.path.join(data_dir.replace("ACDC_volume", "ACDC_data_emb"), initial_frm)
                if self.motion_condition:
                    d["motion_file"] = os.path.join(data_dir.replace("ACDC_volume", "ACDC_motion_field"), d[video_column].replace(".avi", ".npy"))
                    
                d[video_column] = os.path.join(data_dir, d[video_column])
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
        conditions = ["ref_im_emb", "motion_file"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            data["motion_file"] = 1
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
        
class MosmedVolumeDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom",
            motion_condition: bool = False
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.motion_condition = motion_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {"CT-0": 0, "CT-1": 1, "CT-234": 2}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition:
                    initial_frm = d[video_column].replace("_s", "_slice").replace(".avi", ".npy")
                    d["ref_im_emb"] = os.path.join(data_dir.replace("MosMed_volume", "MosMed_data_emb"), initial_frm)
                if self.motion_condition:
                    d["motion_file"] = os.path.join(data_dir.replace("MosMed_volume", "MosMed_motion_field"), d[video_column].replace(".avi", ".npy"))
                    
                d[video_column] = os.path.join(data_dir, d[video_column])
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
        conditions = ["ref_im_emb", "motion_file"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            data["motion_file"] = 1
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
        
class MRNetVolumeDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            width: int = 256,
            height: int = 256,
            keep_all_cond_prob: int = 0.1,
            drop_all_cond_prob: int = 0.1,
            drop_each_cond_prob: list = [0.5, 0.5, 0.5],
            image_condition: bool = False,
            image_encode: str = "Custom",
            motion_condition: bool = False
    ):
        self.data_path = data_path
        self.width = width
        self.height = height
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.image_condition = image_condition
        self.motion_condition = motion_condition
        self.image_encode = image_encode
        
        self.grading_to_id = {0: 0, 1: 1}
        self.data_list = self.load_jsonl(data_path)
    
    def load_jsonl(self, data_dir):
        data_list = []
        with jsonlines.open(data_dir + "/train_metadata.jsonl",'r') as freaders:
            for d in freaders:
                if self.image_condition:
                    initial_frm = d[video_column].replace("_s", "_slice").replace(".avi", ".npy")
                    d["ref_im_emb"] = os.path.join(data_dir.replace("MRNet_volume", "MRNet_data_emb"), initial_frm)
                if self.motion_condition:
                    d["motion_file"] = os.path.join(data_dir.replace("MRNet_volume", "MRNet_motion_field"), d[video_column].replace(".avi", ".npy")) 
                    
                d[video_column] = os.path.join(data_dir, d[video_column])
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
        conditions = ["ref_im_emb", "motion_file"]
        if seed < keep_all_prob:
            results = data
        elif seed < keep_all_prob + drop_all_prob:
            data["ref_im_emb"] = 1
            data["motion_file"] = 1
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
        
# load cond videos
def load_video_frames(video_path):
    video = []
    cap_m = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap_m.read()
        if ret:
            video.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]))
        else:
            break
    video = torch.stack(video).to(memory_format=torch.contiguous_format).float()
    video = video / 127.5 - 1.0
    video = video.unsqueeze(0) # [b f h w c]
    video = rearrange(video, "b f h w c -> b c f h w")
    return video

# load cond ref. img
def load_reference_img(img_path):
    im = cv2.imread(img_path)
    im = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]).to(memory_format=torch.contiguous_format).float() # h, w, c
    im = im / 127.5 - 1.0
    im = rearrange(im[None, :, :, :], "f h w c -> c f h w") # 1, 1, h, w
    im = im.unsqueeze(0) # [b c f h w]
    return im
