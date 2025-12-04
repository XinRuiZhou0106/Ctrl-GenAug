import os, json
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import glob
import torch
import cv2
import shutil
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import imageio
from einops import rearrange
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from diffusers import AutoencoderKL
from argparse import ArgumentParser

import sys
sys.path.append("") # if needed
from Real_Base_Classifiers import mmaction_model
from Real_Base_Classifiers.all_organs.mmaction.utils import (build_dp)

def read_video(syn_p, label):
    frag_frames = []
    cap = imageio.get_reader(syn_p, 'ffmpeg')
    for frame in cap:
        frag_frames.append(frame)
        
    fragment = np.stack(frag_frames,axis=0) 
    fragment = fragment[:,:,:,0] 
    assert fragment.shape == (8, 256, 256) # ensure the correct dimension
    # common operation
    fragment = [(np.array(frag,dtype=np.float32))/255.0 for frag in fragment]
    fragment = [normalize(frag) for frag in fragment] 
    fragment = torch.stack(fragment,dim=0)
    fragment = fragment.permute(3,0,1,2) # c(1), frag_len, 256, 256
    label = torch.from_numpy(np.array(label, dtype=np.int64))
    return fragment.unsqueeze(0), label.unsqueeze(0)

def normalize(im):
    """
    Normalize volume's intensity to range [0, 1], for suing image processing
    Compute global maximum and minimum cause cerebrum is relatively homogeneous
    """
    mean = np.mean(im)
    std = np.std(im)
    if std == 0:
        std = 1
    gray_im = (im - mean) / std
    return torch.from_numpy(gray_im[:, :, np.newaxis])

def kmeans_threshold_inner_video_filter(VAE_Temp_value):
    kmeans = KMeans(n_clusters=4, random_state=3407).fit(np.array(VAE_Temp_value).reshape(-1, 1))
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # threshold
    sorted_centroids = sorted(centroids.flatten())
    low_threshold = (sorted_centroids[0] + sorted_centroids[1]) / 2
    high_threshold = (sorted_centroids[2] + sorted_centroids[3]) / 2
    
    for i in range(4):
        print(f"Cluster {i} size: {np.sum(labels == i)}")
    
    # visualize
    plt.scatter(np.array(VAE_Temp_value), np.zeros_like(np.array(VAE_Temp_value)), c=labels, cmap='viridis')
    plt.axvline(x=low_threshold, color='r', linestyle='--', label='Low threshold')
    plt.axvline(x=high_threshold, color='b', linestyle='--', label='High threshold')
    plt.legend()
    plt.xlabel("Video cross-frame similarity")
    plt.yticks([])
    plt.title("K-means clustering with thresholds for Sequential filtering (inner-video)")
    plt.savefig(f"{syn_data_p}/inner-video-kmeans-threshold.png", bbox_inches='tight', dpi=300)
    
    return low_threshold, high_threshold

def load_frames(video_path):
    video = []
    cap_m = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap_m.read()
        if ret:
            video.append(torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]))
        else:
            break
    video = torch.stack(video)
    video = rearrange(video, "f h w c -> f c h w") 
    return video

def compare_inter_video_similarity(saved_data_set, syn_name, latents, video_length):
    # filter or not
    included = True
    for save_data_set_latents in saved_data_set.values():
        # calculate inter video similarity
        # flatten
        feat1 = save_data_set_latents.reshape(video_length, -1)
        feat2 = latents.reshape(video_length, -1)
        cosine_similarity = F.cosine_similarity(feat1, feat2).mean().item()
        if cosine_similarity >= 98:
            included = False
            break
    if included:
        saved_data_set[syn_name] = latents
    return saved_data_set


if __name__ == "__main__":
    # Script for MosMedData. You can modify the model- and data-related paths to filter your own data.

    parser = ArgumentParser()
    parser.add_argument('--syn_data_p', type=str, required=True, help='synthetic database path')
    parser.add_argument('--cls_model_name', type=str, required=True, help='classifier employed for semantic filtering')
    parser.add_argument('--cls_model_pth', type=str, required=True, help='ckpt of the classifier employed for semantic filtering')
    parser.add_argument('--load_2d_pretrained_model_name', type=str, required=True, help='your pretrained 2d model')
    parser.add_argument('--save_base', type=str, required=True, help='path of the final synthetic databases')
    
    args = parser.parse_args()

    syn_data_p = args.syn_data_p
    load_2d_pretrained_model_name = args.load_2d_pretrained_model_name

    # well-trained classifier in the real data domain
    # In our experiments, the classifier employed for semantic filtering is identical to the downstream classifier by default.
    cls_model_name = args.cls_model_name
    cls_model_pth = args.cls_model_pth
    
    """Class Semantics Misalignment Filtering"""
    print("*********************")
    cls_map_ids = {"CT-0": 0, "CT-1": 1, "CT-234": 2}
    syn_data_info = {}
    for syn_p in glob.glob(f"{syn_data_p}/*.mp4"):
        syn_name = syn_p.split("/")[-1]
        if syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))] in syn_data_info:
            syn_data_info[syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))]]["syn_data"].append(syn_name)
        else:
            syn_data_info[syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))]] = {}
            syn_data_info[syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))]]["syn_data"] = [syn_name]
    
    for syn_name, syn_info in syn_data_info.items():
        cls = syn_info["syn_data"][0].split("_")[-2]
        # check
        for i in range(len(syn_info["syn_data"])):
            assert cls == syn_info["syn_data"][i].split("_")[-2], "label not matches!"
        # add
        syn_info["cls"] = cls_map_ids[cls]
    
    print("Using Real CLS model: ", cls_model_name)
    
    device = torch.device('cuda')
    # you need to first build mmaction environment
    model = mmaction_model(config_path=f"Real_Base_Classifiers/all_organs/mmaction/configs/recognition/{cls_model_name}/{cls_model_name}_r50_32x2x1_100e_kinetics400_rgb.py",
                           checkpoint_path=cls_model_pth)
    model = build_dp(
        model, 'cuda', default_args=dict(device_ids=[0]))
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    
    with tqdm(total=len(syn_data_info), desc="CLS Loss (Noise Filtering)") as pbar:
        for syn_name, syn_info in syn_data_info.items(): # per case
            losses = []
            syn_info["loss_per_video"] = []
            for syn in syn_info["syn_data"]: # per synthetic video
                fragment, label = read_video(syn_p=f"{syn_data_p}/{syn}", label=syn_info["cls"])
                
                # mmaction inference
                fragment, label = fragment.unsqueeze(0), label.unsqueeze(0)
                data = {'imgs': fragment, 'label': label}
                with torch.no_grad():
                    loss = model(return_loss=True, **data)['loss_cls'].item()
                    
                syn_info["loss_per_video"].append(loss)
            syn_info["mean_loss"] = np.mean(syn_info["loss_per_video"])
            pbar.update(1)
    
    # record
    syn_data_info_after_cls_loss = []
    
    # you can modify this parameter to control the filtering strength
    alpha = 1.0 

    for syn_name, syn_info in syn_data_info.items(): # per case
        for idx, syn_avi in enumerate(syn_info["syn_data"]):
            if syn_info["loss_per_video"][idx] <= (syn_info["mean_loss"] * alpha):
                syn_data_info_after_cls_loss.append(syn_avi)
                
    print("After Class Semantics Misalignment Filtering, remaining synthetic videos: ", len(syn_data_info_after_cls_loss))
    
    """Sequential Filtering"""
    print("*********************")
    # model
    video_length = 8
    vae = AutoencoderKL.from_pretrained_mine(f"{load_2d_pretrained_model_name}", subfolder="vae-pretrained").to("cuda")
    vae.eval()
    VAE_Temp_info = json.load(open(f"{syn_data_p}/VAE_Seq_train_results.json"))
    VAE_Temp_value = list(VAE_Temp_info.values())
    
    low_threshold, high_threshold = kmeans_threshold_inner_video_filter(VAE_Temp_value)
    print("Sequential inner-frame filtering threshold: ", [low_threshold, high_threshold])
    
    # Sequential Filtering at inner-sequence level
    syn_data_info_after_cls_loss_Temp1 = []
    for syn_avi, VAE_Temp in VAE_Temp_info.items():
        if (syn_avi in syn_data_info_after_cls_loss) and ((VAE_Temp >= low_threshold) and (VAE_Temp <= high_threshold)):
            syn_data_info_after_cls_loss_Temp1.append(syn_avi)
    print("After Class Semantics Misalignment Filtering and Sequential inner-frame filtering, remaining synthetic videos ", len(syn_data_info_after_cls_loss_Temp1))
    
    # Sequential Filtering at inter-sequence level
    syn_data_info_after_cls_loss_Temp2 = []
    
    after_inner_fitered_data = {}
    for syn_name in syn_data_info_after_cls_loss_Temp1:
        if syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))] in after_inner_fitered_data:
            after_inner_fitered_data[syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))]].append(syn_name)
        else:
            after_inner_fitered_data[syn_name[:syn_name.rfind('_', 0, syn_name.rfind('_'))]] = [syn_name]
            
    with tqdm(total=len(after_inner_fitered_data), desc="Sequential (INTER-FRAME FILTERING)") as pbar:
        for case, syn_names in after_inner_fitered_data.items(): # per case
            saved_data_set = {}
            for syn_name in syn_names:
                syn_video = load_frames(f"{syn_data_p}/{syn_name}")
                # Convert videos to latent space
                pixel_values = (syn_video / 255.0).to(torch.float32).to("cuda") # f c h w
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() # f, 4, 32, 32
                
                if len(saved_data_set) == 0:
                    saved_data_set[syn_name] = latents
                else: 
                    saved_data_set = compare_inter_video_similarity(saved_data_set, syn_name, latents, video_length=8)
                
            syn_data_info_after_cls_loss_Temp2.extend(list(saved_data_set.keys()))
            pbar.update(1)
    print("After CLS noisy filtering, Sequential inner-frame filtering, and Sequential inter-frame filtering, remaining synthetic videos ", len(syn_data_info_after_cls_loss_Temp2))
    
    
    syn_data_info_filtered_final = syn_data_info_after_cls_loss_Temp2
    
    """Save the final synthetic databases"""
    print("*********************")
    save_ = f"Noisy_data_filter/MosMedData/cls_use_{cls_model_name}"
    os.makedirs(save_, exist_ok=True)
    video_count = 0
    video_cls_count = {"CT-0": 0, "CT-1": 0, "CT-234": 0}
    with tqdm(total=len(syn_data_info_filtered_final), desc="SAVING") as pbar_:
        for syn_avi in syn_data_info_filtered_final: 
            video_count += 1
            if "CT-0" in syn_avi:
                video_cls_count["CT-0"] += 1
            elif "CT-1" in syn_avi:
                video_cls_count["CT-1"] += 1
            elif "CT-234" in syn_avi:
                video_cls_count["CT-234"] += 1
            else:
                raise ValueError
                
            # copy from original synthetic databases
            shutil.copy(f"{syn_data_p}/{syn_avi}", f"{save_}/{syn_avi}")
            pbar_.update(1)
    
    print("Synthetic data: ", syn_data_p)
    print("After all filtering, remaining synthetic videos: ", video_count)