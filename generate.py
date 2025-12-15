import torch
import time
import os,math
import imageio, jsonlines, random
from skimage import img_as_ubyte
import numpy as np
from multiprocessing import Pool, get_context

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler

from cond2video.models.unet_3d_multi_modal_input import UNet3DMultiConditionModel
from cond2video.pipelines.pipeline_video import GenVideoPipeline

from einops import rearrange

from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")

random.seed(3407) 

def get_class_syn_names(type, sample_num_each_sam, remaining_sam, data_list_files):
    class_all_syn_names = []
    if sample_num_each_sam != 0:
        print("\n*********************")
        print(f"\nSynthesizing {sample_num_each_sam} guided by each {type} real training clip, respectively (i.e., {sample_num_each_sam} * {len(data_list_files)})")
        # seed -> 66, 67, ... for "each real training clip-guided generation"
        seed = random.sample(list(range(66, 66 + sample_num_each_sam)), sample_num_each_sam)
        for item in data_list_files:
            control_dname = item.split("**")[0].split(".")[0]
            class_all_syn_names.extend([f"{control_dname}_{type}_{s}.mp4" for s in seed])
        if remaining_sam != 0:
            print(f"\nAdditionally select {remaining_sam} {type} real training clips to guide single generation, respectively")
            sampled_data_list_files = random.sample(data_list_files, remaining_sam)
            # seed -> 3407 for "additional real training clip-guided generation"
            seed = 3407
            class_all_syn_names.extend([f"{f.split('**')[0].split('.')[0]}_{type}_{seed}.mp4" for f in sampled_data_list_files])
    else:
        assert remaining_sam != 0
        print("\n*********************")
        print(f"\nSelect {remaining_sam} {type} real training clips to guide single generation, respectively")
        sampled_data_list_files = random.sample(data_list_files, remaining_sam)
        # seed -> 66 (NOTE: seed is set to 3407 only when adopting additional generation)
        seed = 66
        class_all_syn_names.extend([f"{f.split('**')[0].split('.')[0]}_{type}_{seed}.mp4" for f in sampled_data_list_files])
    return class_all_syn_names

def safe_load_npy(path, retries=3, wait=1.0):
    for attempt in range(retries):
        try:
            data = np.load(path, allow_pickle=True)[()]
            return data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                print(f"\n[Error] Failed to load {path} after {retries} retries: {e}", flush=True)
                raise

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m}m{s}s"
    elif m > 0:
        return f"{m}m{s}s"
    else:
        return f"{s}s"

def generate(pipeline,
             grading_to_id,
             user_defined_conditions,  
             device, 
             text_prompts, 
             cls, 
             ref_ims,
             motion_videos,
             trajs,
             output_syn_avi_dir,
             seed
            ):
    
    validation_data = {"video_length": 8,
                       "width": 256,
                       "height": 256,
                       "num_inference_steps": 200,
                       "guidance_scale": 7.5}
    
    # construct batch inputs
    batch_image_hidden_states, batch_motion_hidden_states, batch_class_labels, batch_traj, batch_traj_mask = [], [], [], [], []
    saved_batch_syn_names = []
    for idx, (c, ref_im_p, motion_vi_p, traj_p) in enumerate(zip(cls, ref_ims, motion_videos, trajs)):
        saved_batch_syn_names.append(f"{motion_vi_p.split('/')[-1].split('.')[0]}_{c}_{seed[idx]}.mp4")

        # motion field
        traj_data = safe_load_npy(traj_p)
        traj = traj_data["traj32"].unsqueeze(0)
        traj_mask = traj_data["mask32"].unsqueeze(0)
        batch_traj.append(traj)
        batch_traj_mask.append(traj_mask)

        if "image_condition" in user_defined_conditions:
            image_hidden_states = torch.from_numpy(safe_load_npy(ref_im_p))[None, ...]
            batch_image_hidden_states.append(image_hidden_states)
        
        if "motion_condition" in user_defined_conditions:
            motion_hidden_states = torch.from_numpy(safe_load_npy(motion_vi_p)).transpose(0,1).unsqueeze(0) 
            batch_motion_hidden_states.append(motion_hidden_states)
        
        batch_class_labels.append(torch.LongTensor([[grading_to_id[c]]]))
    
    batch_traj = torch.cat(batch_traj, dim=0).to(device)
    batch_traj_mask = torch.cat(batch_traj_mask, dim=0).to(device)
    batch_class_labels = torch.cat(batch_class_labels, dim=0).to(device)
    batch_image_hidden_states = torch.cat(batch_image_hidden_states, dim=0).to(device) if "image_condition" in user_defined_conditions else None
    batch_motion_hidden_states = torch.cat(batch_motion_hidden_states, dim=0).to(device) if "motion_condition" in user_defined_conditions else None
    
    generator = [torch.Generator(device=device).manual_seed(s) for s in seed]

    # batch generation
    sample = pipeline(text_prompts, 
                      generator=generator,
                      image_hidden_states=batch_image_hidden_states,
                      motion_hidden_states=batch_motion_hidden_states, 
                      class_labels=batch_class_labels,
                      traj=batch_traj, 
                      traj_mask=batch_traj_mask, 
                      **validation_data).videos # tensor(b c f h w)
    
    # ======= save =======
    for i in range(sample.shape[0]):
        # read ims
        res_video = rearrange(sample[i], "c t h w -> t c h w") # one sample contains t frames
        video_frs = []
        for r in res_video:
            r = r.transpose(0, 1).transpose(1, 2)
            r = (r * 255).numpy().astype(np.uint8)
            r = np.repeat(r, 3, axis=2)
            video_frs.append(r)
        
        # make videos (mp4)
        imageio.mimsave(output_syn_avi_dir + f"/{saved_batch_syn_names[i]}", [img_as_ubyte(v) for v in video_frs], fps=2.5)

# Worker
def worker_process(worker_idx, batches_per_worker, shared_params):
    """Each worker initializes the pipeline only once, and then processes batches iteratively"""
    worker_start = time.time()

    (load_2d_pretrained_model_name, load_3d_trained_model_name, user_defined_conditions, grading_to_id, device_id_list, data) = shared_params
    
    # you may modify these lines
    vae = AutoencoderKL.from_pretrained_mine(f"{load_2d_pretrained_model_name}", subfolder="vae-pretrained")
    text_encoder = CLIPTextModel.from_pretrained(f"{load_2d_pretrained_model_name}", subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(f"{load_2d_pretrained_model_name}", subfolder="tokenizer")
    unet = UNet3DMultiConditionModel.from_trained_3d(f"{load_3d_trained_model_name}/checkpoint", subfolder="unet3d")

    device = torch.device(f"cuda:{device_id_list[worker_idx]}")
    
    # Get the validation pipeline
    pipeline = GenVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(load_2d_pretrained_model_name, subfolder="scheduler")
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    # check defined === model driven
    for c in user_defined_conditions:
        assert eval(f'unet.{c}'), f"user defined condition {c} is not compatible with current model!"
    
    for batch_idx, (class_batch_syn_names, type) in enumerate(batches_per_worker):
        start_time = time.time()
        cur_batch_size = len(class_batch_syn_names)
        # conditions in the batch
        batch_text_prompts = [""] * cur_batch_size
        batch_cls = [type] * cur_batch_size
        batch_ref_ims, batch_motion_videos, batch_trajs = [], [], []
        # seed
        batch_seed = []

        for syn_name in class_batch_syn_names: 
            control_dname = syn_name.split(f"_{type}_")[0]
            # please modify the path on your own
            batch_ref_ims.append(f"{data}/MosMed_data_emb/{control_dname.replace('_s', '_slice')}.npy")
            batch_motion_videos.append(f"{data}/MosMed_motion_field/{control_dname}.npy")
            batch_trajs.append(f"{data}/MosMed_motion_field_trajectory/{control_dname}.npy")
            batch_seed.append(int(syn_name.split(f"_{type}_")[1].split(".")[0]))

        # output path
        used_cond = "".join([w[0] for w in user_defined_conditions])
        output_syn_avi_dir = f"{load_3d_trained_model_name}/cond_{used_cond}_synthesis_mp4"
        os.makedirs(output_syn_avi_dir, exist_ok=True)

        # batch generation
        generate(
            pipeline,
            grading_to_id,
            user_defined_conditions,
            device,
            text_prompts=batch_text_prompts,
            cls=batch_cls,
            ref_ims=batch_ref_ims,
            motion_videos=batch_motion_videos,
            trajs=batch_trajs,
            output_syn_avi_dir=output_syn_avi_dir,
            seed=batch_seed
        )
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n[Worker {worker_idx}] Batch {batch_idx+1}/{len(batches_per_worker)} done "
              f"({cur_batch_size} sample, time: {format_time(elapsed)})", flush=True)
    
    del pipeline
    torch.cuda.empty_cache()

    total_elapsed = time.time() - worker_start
    print(f"\n[Worker {worker_idx}] Completed all {len(batches_per_worker)} batches ", 
          f"in {format_time(total_elapsed)}", flush=True)


if __name__ == "__main__":
    # Generation script for MosMedData. You can modify the data paths and conditional control variables to generate your own data.

    parser = ArgumentParser()
    parser.add_argument('--load_2d_pretrained_model_name', type=str, required=True, help='your pretrained 2d model')
    parser.add_argument('--load_3d_trained_model_name', type=str, required=True, help='your pretrained 3d model')
    parser.add_argument('--user_defined_conditions', type=list[str], required=True, help='we use class label by default') # ["image_condition", "motion_condition"]
    parser.add_argument('--predifined_sampled_num', type=int, required=True, help='the number of synthetic data per class') # 1500
    parser.add_argument('--class_label', type=str, default='CT-234')
    
    # Batch generation setting
    parser.add_argument('--batch_size', type=int, required=True) # 1
    parser.add_argument('--num_workers', type=int, required=True) # 3
    parser.add_argument('--device_id_list', type=list[int], required=True) # [0,1,2]

    args = parser.parse_args()

    load_2d_pretrained_model_name = args.load_2d_pretrained_model_name
    load_3d_trained_model_name = args.load_3d_trained_model_name
    
    user_defined_conditions = args.user_defined_conditions
    
    predifined_sampled_num = args.predifined_sampled_num
    batch_size = args.batch_size
    num_workers = args.num_workers
    device_id_list = args.device_id_list
    assert num_workers == len(device_id_list)

    data = "MosMedData"
    grading_to_id = {"CT-0": 0, "CT-1": 1, "CT-234": 2}
    
    # metadata label file
    data_list_dict = {"CT-0": [], "CT-1": [], "CT-234": []}
    with jsonlines.open(f'{data}/MosMed_volume/test_metadata.jsonl','r') as freaders:
        for d in freaders:
            data_list_dict[d["cls"]].append(d["file_name"] + "**" + d["cls"])
    
    print("\nCurrent Inputing Conditions: ", user_defined_conditions)
    
    for type, data_list_files in data_list_dict.items(): # sampling based on grading (cls)
        
        if type == args.class_label:
            
            sample_num_each_sam, remaining_sam = predifined_sampled_num // len(data_list_files), predifined_sampled_num % len(data_list_files)
            
            assert (sample_num_each_sam * (len(data_list_files)) + 1 * remaining_sam) == predifined_sampled_num

            class_all_syn_names = get_class_syn_names(type, sample_num_each_sam, remaining_sam, data_list_files)
            assert len(class_all_syn_names) == predifined_sampled_num

            num_batches = math.ceil(predifined_sampled_num / batch_size)
            
            # all batch info (batch_idx, batch_data, batch_cls)
            all_batch_info = [(class_all_syn_names[i * batch_size: min((i + 1) * batch_size, predifined_sampled_num)], type) for i in range(num_batches)]
            
            # ======= multiprocessing =======
            print("\n*********************")
            print(f"\nStarting multi-process synthesis for {type}: {num_batches} batches, using {num_workers} workers", flush=True)

            worker_batches = [all_batch_info[i::num_workers] for i in range(num_workers)]
            assert sum([len(w_b) for w_b in worker_batches]) == num_batches
            shared_params = (load_2d_pretrained_model_name, load_3d_trained_model_name, user_defined_conditions, grading_to_id, device_id_list, data)

            print("\n****** Starting Generation ******")

            ctx = get_context("spawn")
            pool = ctx.Pool(num_workers)
            try:
                pool.starmap(worker_process, [(i, worker_batches[i], shared_params) for i in range(num_workers)])
            finally:
                pool.close()   
                pool.join()    

            print(f"\nAll {num_batches} batches across {num_workers} workers finished successfully!")
