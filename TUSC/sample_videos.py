import jsonlines
import json
import cv2
import os
import numpy as np
import imageio
import shutil
from skimage import img_as_ubyte
from argparse import ArgumentParser

def adaptive_sampling(fragment, num_clips, frames_per_clip, clip_sampling_interval, after_sampled_num):
    sampled_status = False
    clips = []
    # Before Stage 1, we must ensure that no overlap occurs under the predefined sampling parameter settings
    for n_clip in num_clips:
        available_frames = len(fragment) - (frames_per_clip * n_clip)
        if available_frames >= 0: 
            sampled_status = True
            interval = available_frames // (n_clip - 1) if n_clip > 1 else 0
            break
    if not sampled_status:
        return clips
    
    # Stage 1
    for i in range(n_clip):
        start_id = i * (frames_per_clip + interval)
        clips.append(fragment[start_id: start_id + frames_per_clip])
    
    # Stage 2
    clips = [clip[::clip_sampling_interval+1] for clip in clips]
    
    # check & print
    assert int(clips[-1][-1].split("_")[1].split(".")[0]) <= int(fragment[-1].split("_")[1].split(".")[0])
    sampled_clips_show = []
    for c in clips:
        assert len(c) == after_sampled_num
        c = [int(n.split("_")[1].split(".")[0]) for n in c]
        sampled_clips_show.append(c)
    
    print("***************")
    print(f"{fragment[0].split('_')[0]}: Num-Clip = {n_clip}; Sampled-Interval = {interval}")
    print("Ori Video: ", [int(j.split("_")[1].split(".")[0]) for j in fragment])
    print("Sampled Clips: ", sampled_clips_show)
    
    return clips

if __name__ == "__main__":
    """
    To avoid redundency, we adopt the following uniform sampling strategy.
        stage 1: Uniformly sample multiple clips across the entire sequence.
        stage 2: For each sampled clip (with a length of 'frames_per_clip'), 
                    sample internal frames with a sampling interval of 'clip_sampling_interval'
                    to obtain the final dataset.
    """

    parser = ArgumentParser()
    parser.add_argument('--all_frames_dir', type=str, required=True, help='Directory containing all previously extracted images')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--frames_per_clip', type=int, required=True)
    parser.add_argument('--clip_sampling_interval', type=int, required=True)
    parser.add_argument('--required_clip_num', type=int, required=True)
    args = parser.parse_args()
    
    meta_file = []
    meta_file_sampled_frames = []
    
    all_frames_dir = args.all_frames_dir
    mode = args.mode
    frames_per_clip = args.frames_per_clip # parameter used in the stage 1
    clip_sampling_interval = args.clip_sampling_interval # parameter used in the stage 2
    required_clip_num = args.required_clip_num

    # Load the metadata of all previously extracted frames
    all_videos = {}
    with open(f"TUSC/TUSC_all_images_consecutive/{mode}_metadata.jsonl") as file:
        for ex in jsonlines.Reader(file):
            nodule_name = ex["file_name"][:ex["file_name"].rfind("_")]
            text, cls = ex["text"], ex["cls"]
            if nodule_name not in all_videos: # new
                all_videos[nodule_name] = {}
                all_videos[nodule_name]["text"] = text
                all_videos[nodule_name]["cls"] = cls
                
                all_videos[nodule_name]["frs"] = []
                all_videos[nodule_name]["frs"].append(ex["file_name"])
            else: # old
                all_videos[nodule_name]["frs"].append(ex["file_name"])

    save_frag = "TUSC/TUSC_video"
    save_sampled_frames = "TUSC/TUSC_data"
    os.makedirs(save_frag, exist_ok=True)
    os.makedirs(save_sampled_frames, exist_ok=True)
    
    for nodule_name, info in all_videos.items():
        frames = info["frs"]
        frames = sorted(frames, key=lambda x: int(x[x.rfind("_")+1:].split(".")[0]))
        
        # Sampling
        clips = adaptive_sampling(fragment=frames, 
                                  num_clips=list(range(4, 0, -1)), 
                                  frames_per_clip=frames_per_clip, 
                                  clip_sampling_interval=clip_sampling_interval,
                                  after_sampled_num=required_clip_num) 
        
        if len(clips) == 0:
            continue
        
        # Save the sampled frames/slices for training our VAE and sequence generator (Pretraining Stage)
        for clip in clips:
            for n in clip:
                meta_file_sampled_frames.append({"file_name": n, 
                                                 "text": info["text"],
                                                 "cls": info["cls"]})
                shutil.copy(os.path.join(all_frames_dir, n), os.path.join(save_sampled_frames, n))
        
        # Save the sampled clips for training our sequence generator (Finetuning Stage)      
        for idx, clip in enumerate(clips):
            meta_file.append({"file_name": nodule_name + f"_f{clip[0].split('_')[1].split('.')[0]}.avi", 
                              "text": info["text"],
                              "cls": info["cls"]})
        fps = 2.5
        for idx, clip in enumerate(clips):
            video_frs = []
            for n in clip:
                video_frs.append(cv2.imread(os.path.join(all_frames_dir, n)))
            # make videos
            videoWriter = cv2.VideoWriter(save_frag + f"/{nodule_name}_f{clip[0].split('_')[1].split('.')[0]}.avi", cv2.VideoWriter_fourcc(*'I420'), fps, (256, 256))
            for i in range(len(video_frs)):
                videoWriter.write(video_frs[i])
                cv2.waitKey(50)
            videoWriter.release()

    # statistics
    print(f"{mode} set video num: {len(meta_file)}")
    cls_stat = [l["cls"] for l in meta_file]
    level, count = np.unique(cls_stat, return_counts=True)
    print("Videos TI-RADS level count: ", dict(zip(level, count)))

    # Image-level metadata file for LDM pretraining
    with open(f"{save_sampled_frames}/{mode}_metadata.jsonl", 'w') as f_new:
        for item in meta_file_sampled_frames:
            json.dump(item, f_new)
            f_new.write('\n')

    # Clip-level metadata file for Sequence LDM finetuning
    with open(f"{save_frag}/{mode}_metadata.jsonl", 'w') as f_new:
        for item in meta_file:
            json.dump(item, f_new)
            f_new.write('\n')
    
    