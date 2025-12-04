import cv2
import glob
import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from diffusers import AutoencoderKL
from tqdm import tqdm
import imageio
import json
from argparse import ArgumentParser

# load video frames
def load_frames(video_path):
    video = []
    cap_m = imageio.get_reader(video_path, 'ffmpeg')
    for frame in cap_m:
        video.append(torch.from_numpy(frame[:, :, 0][..., None]))
        
    video = torch.stack(video)
    video = rearrange(video, "f h w c -> f c h w")
    return video

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--load_2d_pretrained_model_name', type=str, required=True, help='your pretrained 2d model')
    parser.add_argument('--syn_data_p', type=str, required=True, help='synthetic database path')
    
    args = parser.parse_args()
    
    # model
    load_2d_pretrained_model_name = args.load_2d_pretrained_model_name
    vae = AutoencoderKL.from_pretrained_mine(f"{load_2d_pretrained_model_name}", subfolder="vae-pretrained").to("cuda:0")
    vae.eval()
    
    # data
    syn_videos = []
    syn_video_names = []
    syn_data_p = args.syn_data_p
    for syn_avi_p in glob.glob(f"{syn_data_p}/*.mp4"):
        syn_videos.append(load_frames(syn_avi_p))
        syn_video_names.append(syn_avi_p.split("/")[-1])
    syn_videos = torch.stack(syn_videos)
    
    # Convert videos to latent space
    pixel_values = (syn_videos / 255.0).to(torch.float32).to("cuda:0") # b f c h w
    video_length = pixel_values.shape[1]
    with torch.no_grad():
        latents = []
        with tqdm(total=pixel_values.shape[0], desc="Computing VAE embeddings...") as pbar_:
            for pixel_value_sample in pixel_values: # f c h w
                latent_sample = vae.encode(pixel_value_sample).latent_dist.sample() # f, 4, 32, 32
                latents.append(latent_sample)
                pbar_.update(1)
            pbar_.close()
        latents = torch.stack(latents)
    
    # compute VAE-Seq
    VAE_Seq_all_videos = []
    VAE_Seq_info = {}
    with tqdm(total=latents.shape[0], desc="Computing scores...") as pbar:
        for i in range(latents.shape[0]): # each synthetic video
            video_latent = latents[i]
            # consecutive frames
            feat1_idx = torch.arange(0, video_length-1)
            feat2_idx = torch.arange(1, video_length)
            
            feat1, feat2 = video_latent[feat1_idx], video_latent[feat2_idx]
            # flatten
            feat1 = feat1.reshape(video_length-1, -1)
            feat2 = feat2.reshape(video_length-1, -1)
            
            cosine_similarity = F.cosine_similarity(feat1, feat2).mean().item()
            # print(cosine_similarity) # a video
            VAE_Seq_all_videos.append(cosine_similarity)
            VAE_Seq_info[syn_video_names[i]] = round(cosine_similarity * 100, 2)
            pbar.update(1)
        pbar.close()
    
    print("model: ", syn_data_p)
    print("VAE-Seq: ", round(np.mean(VAE_Seq_all_videos) * 100, 2))
    
    with open(f"{syn_data_p}/VAE_Seq_train_results.json", "w", encoding="utf-8") as f:
        json.dump(VAE_Seq_info, f, indent=4)
        
    