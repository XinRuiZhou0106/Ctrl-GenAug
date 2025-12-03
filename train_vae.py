import argparse
import logging
import math
import os
import cv2
import inspect

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from typing import Dict, Optional
from omegaconf import OmegaConf
from tqdm.auto import tqdm


import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from cond2img.data.train_vae_dataset import MosMedDataset, MRNetDataset, ACDCDataset, ThyroidDataset

dataset = {"MosMed": MosMedDataset, "MRNet": MRNetDataset, "ACDC": ACDCDataset, "TUSC": ThyroidDataset}

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, test_dataloader, accelerator, iter, save_p, epoch):
    logger.info("Running validation... ")
    vae.eval()

    progress_bar = tqdm(range(len(test_dataloader)), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    test_rec_loss = 0.
    for step, batch in enumerate(test_dataloader):
        with accelerator.accumulate(vae) and torch.no_grad():
            # perform vae
            out, loss, loss_log = vae(batch["pixel_values"], sample_posterior=True)

        # Checks if the accelerator has performed an optimization step behind the scenes
        if accelerator.sync_gradients:
            progress_bar.update(1)

        logs = {"step_loss": loss.detach().item()}
        logs.update(loss_log)
        progress_bar.set_postfix(**logs)
        test_rec_loss += loss_log["rec_loss"]

        ori = np.stack(batch["images"])[..., np.newaxis]
        pred = (out.sample / 2 + 0.5).clamp(0, 1) # denormalize
        pred = pred.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        pred = (pred * 255).round().astype("uint8") 

        res = np.concatenate([ori, pred], 1)
        res = np.repeat(res, 3, axis=3)
        init = res.copy()[0]
        init = cv2.putText(init, 'Ori', (10,30), cv2.FONT_HERSHEY_TRIPLEX, 1.0, [0, 255, 255], 2)
        init = cv2.putText(init, 'Rec', (10,286), cv2.FONT_HERSHEY_TRIPLEX, 1.0, [0, 255, 255], 2)
        for i in range(1, pred.shape[0]):
            init = np.concatenate([init, res[i]], 1)

        cv2.imwrite(f"{save_p}/iter_{iter}_step_{step}.png", init)
    
    test_rec_loss = test_rec_loss / (step + 1)
    with open(save_p.split("/inter_test_results")[0] + "/test_rec_loss.txt", "a+", encoding="utf-8") as f:
        f.write(f"Iter{iter}-Epo{epoch}:   {test_rec_loss}\n")
    f.close()

def main(
    data_name: str,
    pretrained_model_path: str,
    image_column: str,
    output_dir: str,
    train_data: Dict,
    test_data: Dict,
    validation_steps: int = 100,
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
    
    # Train based on pretrained ckpt from hugging face (https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main/vae)
    vae = AutoencoderKL.from_pretrained_mine(config_path=pretrained_model_path, subfolder="vae")

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "vae"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()
    accelerator.register_save_state_pre_hook(save_model_hook)

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_dataset = dataset[data_name](**train_data)
    test_dataset = dataset[data_name](**test_data)

    # Preprocessing the dataset
    def collate_fn(examples):
        # image preprocess
        images = [cv2.imread(ex[image_column], 0) for ex in examples]
        for ex, image in zip(examples, images):
            im = (image / 127.5 - 1.0).astype(np.float32)
            ex["pixel_values"] = torch.from_numpy(im[:, :, np.newaxis]).contiguous().permute(2,0,1)

        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "images": images}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, optimizer, train_dataloader, lr_scheduler, test_dataloader = accelerator.prepare(
        vae, optimizer, train_dataloader, lr_scheduler, test_dataloader
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("vae")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Train Num examples = {len(train_dataset)}")
    logger.info(f"  Test Num examples = {len(test_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
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

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        vae.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step.
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(vae):
                # perform vae
                out, loss, loss_log = vae(batch["pixel_values"], sample_posterior=True)
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        save_p = output_dir + "/inter_test_results"
                        os.makedirs(save_p, exist_ok=True)
                        log_validation(
                        vae,
                        test_dataloader,
                        accelerator,
                        global_step,
                        save_p,
                        epoch
                    )

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs.update(loss_log)
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
