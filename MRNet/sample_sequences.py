import jsonlines
import json
import cv2
import os
import numpy as np
import imageio
import shutil
from skimage.util import img_as_ubyte
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
        print("!!!***************!!!")
        print(f"{fragment[0].split('_slice')[0]}: Need zero-padding because of total {len(fragment)} slices")
        return [fragment]
    
    # Stage 1
    for i in range(n_clip):
        start_id = i * (frames_per_clip + interval)
        clips.append(fragment[start_id: start_id + frames_per_clip])
    
    # Stage 2
    clips = [clip[::clip_sampling_interval+1] for clip in clips]
    
    # print & check
    assert int(clips[-1][-1].split("_slice")[1].split(".")[0]) <= int(fragment[-1].split("_slice")[1].split(".")[0])
    sampled_clips_show = []
    for c in clips:
        assert len(c) == after_sampled_num
        c = [int(n.split("_slice")[1].split(".")[0]) for n in c]
        sampled_clips_show.append(c)
    
    print("***************")
    print(f"{fragment[0].split('_slice')[0]}: Num-Clip = {n_clip}; Sampled-Interval = {interval}")
    print("Ori Video: ", [int(j.split("_slice")[1].split(".")[0]) for j in fragment])
    print("Sampled Clips: ", sampled_clips_show)
    
    return clips

if __name__ == "__main__":
    """
    To avoid redundency, we adopt the following uniform sampling strategy.
        stage 1: Uniformly sample multiple clips across the entire sequence.
        stage 2: For each sampled clip (with a length of 'frames_per_clip'), 
                    sample internal frames/slices with a sampling interval of 'clip_sampling_interval'
                    to obtain the final dataset.
    """

    parser = ArgumentParser()
    parser.add_argument('--all_slices_dir', type=str, required=True, help='Directory containing all previously extracted slices')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True)
    parser.add_argument('--frames_per_clip', type=int, required=True)
    parser.add_argument('--clip_sampling_interval', type=int, required=True)
    parser.add_argument('--required_clip_num', type=int, required=True)
    args = parser.parse_args()

    meta_file = []
    meta_file_sampled_slices = []

    all_slices_dir = args.all_slices_dir
    mode = args.mode
    frames_per_clip = args.frames_per_clip # parameter used in the stage 1
    clip_sampling_interval = args.clip_sampling_interval # parameter used in the stage 2
    required_clip_num = args.required_clip_num

    # Load the metadata of all previously extracted slices
    all_volumes = {}
    with open(f"MRNet/MRNet_all_slices/{mode}_metadata.jsonl") as file:
        for ex in jsonlines.Reader(file):
            volume_name = ex["file_name"][:ex["file_name"].rfind("_")]
            cls = ex["cls"]
            if volume_name not in all_volumes: # new
                all_volumes[volume_name] = {}
                all_volumes[volume_name]["cls"] = cls
                
                all_volumes[volume_name]["frs"] = []
                all_volumes[volume_name]["frs"].append(ex["file_name"])
            else: # old
                all_volumes[volume_name]["frs"].append(ex["file_name"])

    save_frag = "MRNet/MRNet_volume"
    save_sampled_slices = "MRNet/MRNet_data" 
    os.makedirs(save_frag, exist_ok=True)
    os.makedirs(save_sampled_slices, exist_ok=True)

    for volume_name, info in all_volumes.items():
        frames = info["frs"]
        frames = sorted(frames, key=lambda x: int(x[x.rfind("_")+6:].split(".")[0]))
        
        # Sampling
        clips = adaptive_sampling(fragment=frames, 
                                  num_clips=list(range(10, 0, -1)), 
                                  frames_per_clip=frames_per_clip, 
                                  clip_sampling_interval=clip_sampling_interval,
                                  after_sampled_num=required_clip_num)  
        
        if len(clips) == 0:
            raise ValueError
        
        # Save the sampled frames/slices for training our VAE and sequence generator (Pretraining Stage)
        for clip in clips:
            for n in clip:
                meta_file_sampled_slices.append({"file_name": n, 
                                                 "cls": info["cls"]})
                shutil.copy(os.path.join(all_slices_dir, n), os.path.join(save_sampled_slices, n))
        
        # Save the sampled clips for training our sequence generator (Finetuning Stage)
        for idx, clip in enumerate(clips):
            meta_file.append({"file_name": volume_name + f"_s{clip[0].split('_slice')[1].split('.')[0]}.avi",
                              "cls": info["cls"]})
        fps = 2.5
        for idx, clip in enumerate(clips):
            video_frs = []
            for n in clip:
                video_frs.append(cv2.imread(os.path.join(all_slices_dir, n)))
            if len(video_frs) < required_clip_num: # zero-padding
                pad_slices = [np.zeros((256, 256, 3), dtype=np.uint8) for _ in range(required_clip_num - len(video_frs))]
                video_frs.extend(pad_slices)
            assert len(video_frs) == required_clip_num
            # make videos
            videoWriter = cv2.VideoWriter(save_frag + f"/{volume_name}_s{clip[0].split('_slice')[1].split('.')[0]}.avi", cv2.VideoWriter_fourcc(*'I420'), fps, (256, 256))
            for i in range(len(video_frs)):
                videoWriter.write(video_frs[i])
                cv2.waitKey(50)
            videoWriter.release()

    # statistics
    print(f"{mode} set volume num: {len(meta_file)}")
    cls_stat = [l["cls"] for l in meta_file]
    level, count = np.unique(cls_stat, return_counts=True)
    print("Volumes class count: ", dict(zip(level, count)))

    # Slice-level metadata file for LDM pretraining
    with open(f"{save_sampled_slices}/{mode}_metadata.jsonl", 'w') as f_new:
        for item in meta_file_sampled_slices:
            json.dump(item, f_new)
            f_new.write('\n')

    # Clip-level metadata file for Sequence LDM finetuning
    with open(f"{save_frag}/{mode}_metadata.jsonl", 'w') as f_new:
        for item in meta_file:
            json.dump(item, f_new)
            f_new.write('\n')
    
    