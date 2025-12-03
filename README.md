# Ctrl-GenAug
This codebase provides the official PyTorch implementation for Ctrl-GenAug: Controllable Generative Augmentation for Medical Sequence Classification (Under Review)

[![Ctrl-GenAug-previous-version](https://img.shields.io/badge/Previous%20Version-arXiv-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/pdf/2409.17091)

## 📖 Introduction
In this work, we present *Ctrl-GenAug*, a novel and general generative augmentation framework that enables highly semantic- and sequential-customized sequence synthesis and suppresses incorrectly synthesized samples, to aid **medical sequence classification**. Specifically, we first design a **multimodal conditions-guided sequence generator** for controllably synthesizing diagnosis-promotive samples. A sequential augmentation module is integrated to enhance the temporal/stereoscopic coherence of generated samples. Then, we propose a **noisy synthetic data filter** to suppress unreliable cases at the semantic and sequential levels. Extensive experiments on 5 medical datasets with 4 different modalities, including comparisons with 15 augmentation methods and evaluations using 11 networks trained on 3 paradigms, comprehensively demonstrate the effectiveness and generality of *Ctrl-GenAug*, particularly with pronounced performance gains in underrepresented high-risk populations and out-domain conditions.

## :mega: Overall Framework
#### Pipeline of using our *Ctrl-GenAug* to facilitate medical sequence recognition, which can be worked with a variety of classifiers:

<p align="center">
  <img src="assets/images/overall_framework.png">
</p>

## 💫 Sequence Generator
#### Our generator produces a real-domain style sequence that faithfully adheres to all specified conditions.

<p align="center">
  <img src="assets/images/sequence_generator.png">
</p>

## 🕹️ Preparations

### 1. Installation

Requirements:
- Python==3.8.13
- torch==1.12.1+cu116
- torchvision==0.13.1+cu116
- transformers==4.34.1
- tokenizers==0.14.1
- ffmpeg
- motion-vector-extractor==1.0.6

You can also create an identical environment to ours using the following command:
```
cd ./Ctrl-GenAug
conda env create -f environment.yaml
```

### 2. Download the Public Real Datasets

- [MosMedData](https://www.kaggle.com/datasets/mathurinache/mosmeddata-chest-ct-scans-with-covid19)
- [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/)
- [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- [TUSC](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5)

### 3. Pre-process the Real Datasets

We provide the preprocessing scripts along with our dataset split information. Please follow the steps below.

#### Step (a) Extract all slices/frames and save their metadata, including file names and labels:

```bash
# For MosMedData
python MosMedData/extract_all_slices.py
```

#### Step (b) Perform uniform sampling based on the data extracted in step (a)

```bash
# For MosMedData
python MosMedData/sample_sequences.py \
    --all_slices_dir <> \          # Directory containing all slices extracted in step (a)
    --mode <> \                    # Train / test only
    --frames_per_clip 15 \         # See the script for detailed parameter descriptions
    --clip_sampling_interval 1 \   # See the script for detailed parameter descriptions
    --required_clip_num 8
```

**Note:**

After completing step (b),
- You can obtain the sampled frames/slices (e.g., ``MosMedData/MosMed_data``) and corresponding metadata labels for training our VAE and sequence generator (**Pretraining Stage**).
- You can accordingly obtain the sampled clips (e.g., ``MosMedData/MosMed_volume``) and corresponding metadata labels for training our sequence generator (**Finetuning Stage**).
- You may delete the folder ``all_slices_dir``, which is no longer needed, to free storage space.

We have also provided metadata labels based on our dataset split for your reference. Taking *MosMedData* as an example:

```
# Slice-level metadata for LDM pretraining
MosMedData/MosMed_data/train_metadata.jsonl
MosMedData/MosMed_data/test_metadata.jsonl

# Clip-level metadata for Sequence LDM finetuning
MosMedData/MosMed_volume/train_metadata.jsonl
MosMedData/MosMed_volume/test_metadata.jsonl
```

#### Step (c) Generate domain-specific image prior features

We extract image prior features for each training frame/slice (e.g., training images in ``MosMedData/MosMed_data``) using the [MedSAM](https://github.com/bowang-lab/MedSAM) image encoder ([base](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) model). Please refer to their instructions.
The resulting image priors for *MosMedData* should be saved in the following form; the same applies to other datasets.

```
Ctrl-GenAug
└── MosMedData
    └── MosMed_data_emb
        ├── study_0001_slice01.npy
        ├── study_0001_slice03.npy
        ├── study_0001_slice05.npy
        └── ...
```

#### Step (d) Produce motion fields and sample motion field-based trajectories

```bash
python extract_motion_field_trajs.py --clip_data_dir <>   # Directory containing the pre-processed clips from step (b)
```

#### Step (e) Arrange all pre-processed data into the following structure to prepare for sequence generator training

```
Ctrl-GenAug
├── MosMedData
│   ├── MosMed_data
│   │   ├── study_0001_slice01.png
│   │   ├── study_0001_slice03.png
│   │   └── ...
│   ├── MosMed_data_emb
│   │   ├── study_0001_slice01.npy
│   │   ├── study_0001_slice03.npy
│   │   └── ...
│   ├── MosMed_volume
│   │   ├── study_0001_s01.avi
│   │   ├── study_0001_s29.avi
│   │   └── ...
│   ├── MosMed_motion_field
│   │   ├── study_0001_s01.npy
│   │   ├── study_0001_s29.npy
│   │   └── ...
│   └── MosMed_motion_field_trajectory
│       ├── study_0001_s01.npy
│       ├── study_0001_s29.npy
│       └── ...
├── MRNet
│   └── ...
├── ACDC
│   └── ...
└── TUSC
    └── ...
```

🥳 After completing the preparations, you can proceed to build the *Ctrl-GenAug* framework.

## 🚀 Step 1: Sequence Generator Training

#### 1. VAE model


## Diagnosis-promotive Synthetic Datasets

To support research in medical sequence analysis, we released the synthetic databases generated by *Ctrl-GenAug*. We hope these resources could serve as a valuable supplement for model development.

- [Download MosMedData-Synthetic (Lung, CT)](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xzhoucv_connect_ust_hk/IQCh5xuYEvc2Qp0MKGCNvY9gAQBxh5XfFKbtjJsV2Ioet2o?e=d3r1uP)
- [Download MRNet-Synthetic (Knee, T2-weighted MRI)](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xzhoucv_connect_ust_hk/IQB9bGIrRc3STL33Fn5O6NiZAf5YE5bXLebibG1A_BU1QYo?e=nuHeGw)
- [Download ACDC-Synthetic (Heart, Cine-MRI)](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xzhoucv_connect_ust_hk/IQCLXd3SoF65SqtgYX7cYqCIAfDGrEn1ExZTTo9hVONMv_o?e=jbKdOw)
- [Download TUSC-Synthetic (Thyroid, Ultrasound)](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xzhoucv_connect_ust_hk/IQCmeFizQ-dJS7fthEr7jXHcAZz2AIEKZMQGnzcwVXJAFss?e=tPLb0w)
- [Download Carotid-Synthetic (Carotid artery, Ultrasound)](https://hkustconnect-my.sharepoint.com/:u:/g/personal/xzhoucv_connect_ust_hk/IQAhXmrsTIiKRqbsc_NAJc8eASEqrOgJVjW2658vyY856Fw?e=qjs8u5)

## :black_nib: Citation

If you find our repo useful for your research, please consider citing our paper:
```bibtex
@article{zhou2024ctrl,
  title={Ctrl-GenAug: Controllable generative augmentation for medical sequence classification},
  author={Zhou, Xinrui and Huang, Yuhao and Dou, Haoran and Chen, Shijing and Chang, Ao and Liu, Jia and Long, Weiran and Zheng, Jian and Xu, Erjiao and Ren, Jie and others},
  journal={arXiv preprint arXiv:2409.17091},
  year={2024}
}
```

