import argparse
import datetime
import random
import logging
import inspect
import math
import os
import cv2
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import imageio
import torch
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from cond2video.models.unet_3d_multi_modal_input import UNet3DMultiConditionModel
from cond2video.data.multi_cond_dataset import MosmedVolumeDataset, MRNetVolumeDataset, ACDCVolumeDataset, ThyroidVideoDataset
from cond2video.pipelines.pipeline_video import GenVideoPipeline
from einops import rearrange

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=16, fps=2.5):
    video = rearrange(videos[0], "c t h w -> t c h w") # one sample contains t frames

    x = torchvision.utils.make_grid(video, nrow=n_rows, padding=5, pad_value=1.0)
    x = x.transpose(0, 1).transpose(1, 2)[:, :, 0]
    if rescale:
        x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    x = (x * 255).numpy().astype(np.uint8)

    imageio.imsave(path, x)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

dataset = {"MosMed": MosmedVolumeDataset, "MRNet": MRNetVolumeDataset, "ACDC": ACDCVolumeDataset, "TUSC": ThyroidVideoDataset}

dataset_grading_to_id = {"MosMed": {"CT-0": 0, "CT-1": 1, "CT-234": 2}, 
                         "MRNet":  {0: 0, 1: 1}, 
                         "ACDC":   {"NOR": 0, "MINF": 1, "DCM": 2, "HCM": 3, "RV": 4}, 
                         "TUSC":   {"TR2-3": 0, "TR4": 1, "TR5": 2}}

def log_validation(data_name, vae, text_encoder, tokenizer, unet, accelerator, epoch, pretrained_2d_model_path, seed, validation_data, condition_flag, output_dir):
    logger.info("Running validation... ")

    # Get the validation pipeline
    pipeline = GenVideoPipeline(
        vae=accelerator.unwrap_model(vae), text_encoder=accelerator.unwrap_model(text_encoder), tokenizer=tokenizer, unet=accelerator.unwrap_model(unet),
        scheduler=DDIMScheduler.from_pretrained(pretrained_2d_model_path, subfolder="scheduler")
    )
    
    pipeline.enable_vae_slicing()
    pipeline.set_progress_bar_config(disable=True)
    
    grading_to_id = dataset_grading_to_id[data_name]
    
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(seed)

    save_inter = output_dir + f"/inter_test_results/iter_{epoch}"
    os.makedirs(save_inter, exist_ok=True)
    
    if condition_flag["image_condition"] and condition_flag["motion_condition"]:
        for labels, vi_p in zip(validation_data.labels, validation_data.videos):
            name = vi_p.split('/')[-1].split('.')[0]
            
            if accelerator.unwrap_model(unet).self_attn_mode == "SAM":
                traj = np.load(vi_p.replace(f"{data_name}_volume", f"{data_name}_motion_field_trajectory").replace(".avi", ".npy"), allow_pickle=True)[()]["traj32"].unsqueeze(0).to(accelerator.device)
                traj_mask = np.load(vi_p.replace(f"{data_name}_volume", f"{data_name}_motion_field_trajectory").replace(".avi", ".npy"), allow_pickle=True)[()]["mask32"].unsqueeze(0).to(accelerator.device)
            else:
                raise ValueError
            
            if condition_flag["image_encode"] == "MedSAM":
                initial_frm = name.replace("_s", "_slice") + ".npy" # something different here if you are running TUSC (please refer to multi_cond_dataset.py)
                ref_im_emb = os.path.join(f"{data_name}/{data_name}_data_emb", initial_frm)
                ref_im_emb = torch.from_numpy(np.load(ref_im_emb, allow_pickle=True)[()])[None, ...]

            motion_vi_p = vi_p.replace(f"{data_name}_volume", f"{data_name}_motion_field").replace(".avi", ".npy")
            motion_hidden_states = torch.from_numpy(np.load(motion_vi_p, allow_pickle=True)[()]).transpose(0,1).unsqueeze(0)

            class_labels = torch.LongTensor([[grading_to_id[labels]]])
            
            sample = pipeline("", generator=generator,
                              image_hidden_states=ref_im_emb.to(accelerator.device), 
                              motion_hidden_states=motion_hidden_states.to(accelerator.device), 
                              class_labels=class_labels.to(accelerator.device), 
                              traj=traj, 
                              traj_mask=traj_mask, 
                              **validation_data).videos # tensor(b c f h w)
            save_videos_grid(sample, f"{save_inter}/{name}.png")

    logger.info(f"Saved samples to {save_inter}")

    del pipeline
    torch.cuda.empty_cache()

def main(
    data_name: str,
    pretrained_2d_model_path: str,
    output_dir: str,
    caption_column: str,
    video_column: str,
    image_condition: bool,
    image_encode: str,
    motion_condition: bool,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = None,
    train_batch_size: int = 1,
    num_train_epochs: int = 1000,
    num_workers: int = 0,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = None,
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
    seed: Optional[int] = None,
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # basic information about code used currently (related to config)
    logger.info("***** Running Configuration Checking *****")
    logger.info(f"  Pretrained 2D-model path = {pretrained_2d_model_path}")
    logger.info(f"  Output path = {output_dir}")
    logger.info(f"  Train data path = {train_data.data_path}")
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models from your trained LDM.
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_2d_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_2d_model_path, subfolder="tokenizer") 
    text_encoder = CLIPTextModel.from_pretrained(pretrained_2d_model_path, subfolder="text_encoder") 
    vae = AutoencoderKL.from_pretrained_mine(pretrained_2d_model_path, subfolder=f"{data_name}-vae-pretrained")
    unet = UNet3DMultiConditionModel.from_pretrained_2d(pretrained_2d_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if any(tr_m in name for tr_m in trainable_modules):
            if accelerator.is_main_process:
                print("trainable module: ", name)
            for params in module.parameters():
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet3d"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
    accelerator.register_save_state_pre_hook(save_model_hook)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_dataset = dataset[data_name](image_condition=image_condition, motion_condition=motion_condition, 
                                       image_encode=image_encode, **train_data)

    # Preprocessing the dataset
    # We need to tokenize input captions.
    def tokenize_captions(example, is_train=True):
        captions = []
        caption = "" 
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
        
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids[0]
    
    # load video frames
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

    def collate_fn(examples):
        # video preprocess
        videos = [load_frames(ex[video_column]) for ex in examples]
        
        if accelerator.unwrap_model(unet).self_attn_mode == "SAM": 
            trajectories = {}
            tr, ma = [], []
            for ex in examples:
                tr.append(np.load(ex[video_column].replace(f"{data_name}_volume", f"{data_name}_motion_field_trajectory").replace(".avi", ".npy"), allow_pickle=True)[()]["traj32"]) 
                ma.append(np.load(ex[video_column].replace(f"{data_name}_volume", f"{data_name}_motion_field_trajectory").replace(".avi", ".npy"), allow_pickle=True)[()]["mask32"])
            trajectories["traj32"], trajectories["mask32"] = torch.stack(tr), torch.stack(ma)
                
        if image_condition:
            if image_encode == "MedSAM":
                ref_images = [torch.from_numpy(np.load(ex["ref_im_emb"], allow_pickle=True)[()]) if ex["ref_im_emb"] !=1 else torch.zeros((256, 64, 64)) for ex in examples]
    
        if motion_condition:
            motion_images = [torch.from_numpy(np.load(ex["motion_file"], allow_pickle=True)[()]).transpose(0,1) if ex["motion_file"] !=1 else torch.zeros((2, 7, 256, 256)) for ex in examples] # [sam1, sam2, ...]. each sam -> (c, f-1, h, w)
            
        if image_condition and motion_condition:
            for ex, video, video_ref_img, video_motion in zip(examples, videos, ref_images, motion_images):
                ex["pixel_values"] = video / 127.5 - 1.0
                
                ex["motion_control"] = video_motion
                ex["input_ids"] = tokenize_captions(ex)
                ex["class_labels"] = torch.LongTensor([ex["cls"]])

            pixel_values = torch.stack([example["pixel_values"] for example in examples])
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            input_ids = torch.stack([example["input_ids"] for example in examples])
            class_labels = torch.stack([example["class_labels"] for example in examples]) # bts, 1
            
            if image_encode == "MedSAM":
                ref_img_control = torch.stack(ref_images)
                ref_img_control = ref_img_control.to(memory_format=torch.contiguous_format).float()
            motion_control = torch.stack([example["motion_control"] for example in examples])
            motion_control = motion_control.to(memory_format=torch.contiguous_format).float()

            if accelerator.unwrap_model(unet).self_attn_mode == "SAM":
                return {"pixel_values": pixel_values, "input_ids": input_ids, "class_labels": class_labels,
                        "ref_img_control": ref_img_control, "motion_control": motion_control, "motion_trajectories": trajectories}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("cond2video")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    assert image_condition == accelerator.unwrap_model(unet).image_condition
    assert motion_condition == accelerator.unwrap_model(unet).motion_condition
    logger.info("***** Multi-modal Condition Configuration *****")
    logger.info(f"  Image Condition = {accelerator.unwrap_model(unet).image_condition}")
    logger.info(f"  Motion Condition = {accelerator.unwrap_model(unet).motion_condition}")
    logger.info(f"  (FOR IMAGE) Encoder = {accelerator.unwrap_model(unet).image_encode}")
    logger.info(f"  (FOR IMAGE) Attention Condition Insert Block = {accelerator.unwrap_model(unet).condition_insert_block}")
    logger.info(f"  (Joint Training Strategy) Keep All Cond Prob = {train_dataset.keep_all_cond_prob}")
    logger.info(f"  (Joint Training Strategy) Drop All Cond Prob = {train_dataset.drop_all_cond_prob}")
    logger.info(f"  (Joint Training Strategy) Drop Each Cond Prob = {train_dataset.drop_each_cond_prob}")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Num Workers = {num_workers}")
    logger.info(f"  Constant lr = {learning_rate}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Checkpointing steps = {checkpointing_steps}")
    logger.info(f"  Validation steps = {validation_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample() # bs*t, 4, 32, 32
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0] # bs, 77, 768
                
                # Get the class labels
                class_labels = batch["class_labels"]

                # Get the cond embedding
                if image_encode != "MedSAM":
                    image_hidden_states = batch["ref_img_control"].to(dtype=weight_dtype) if image_condition else None
                else:
                    image_hidden_states = batch["ref_img_control"] if image_condition else None
                motion_hidden_states = batch["motion_control"].to(dtype=weight_dtype) if motion_condition else None

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")
                
                # for motion field attention
                traj = batch["motion_trajectories"]["traj32"] if accelerator.unwrap_model(unet).self_attn_mode == "SAM" else None
                traj_mask = batch["motion_trajectories"]["mask32"] if accelerator.unwrap_model(unet).self_attn_mode == "SAM" else None

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states,
                                  image_cond=image_hidden_states, motion_cond=motion_hidden_states,
                                  class_labels=class_labels, traj=traj, traj_mask=traj_mask).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        log_validation(
                        data_name, 
                        vae,
                        text_encoder,
                        tokenizer,
                        unet,
                        accelerator,
                        global_step,
                        pretrained_2d_model_path,
                        seed,
                        validation_data,
                        condition_flag={"image_condition": image_condition, 
                                        "motion_condition": motion_condition, 
                                        "image_encode": image_encode},
                        output_dir=output_dir
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if accelerator.is_main_process:
                with open(output_dir + "/train_loss.txt", "a+", encoding="utf-8") as f: 
                    f.write(f"Iter{global_step}-Epo{epoch}:   {loss.detach().item()}\n")
                f.close()

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
