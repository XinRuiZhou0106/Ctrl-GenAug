# Ctrl-GenAug
This codebase provides the official PyTorch implementation for Ctrl-GenAug: Controllable Generative Augmentation for Medical Sequence Classification

## 📖 Introduction
In this work, we present *Ctrl-GenAug*, a novel and general generative augmentation framework that enables highly semantic- and sequential-customized sequence synthesis and suppresses incorrectly synthesized samples, to aid medical sequence classification. Specifically, we first design a **multimodal conditions-guided sequence generator** for controllably synthesizing diagnosis-promotive samples. A sequential augmentation module is integrated to enhance the temporal/stereoscopic coherence of generated samples. Then, we propose a **noisy synthetic data filter** to suppress unreliable cases at the semantic and sequential levels. Extensive experiments on 5 medical datasets with 4 different modalities, including comparisons with 15 augmentation methods and evaluations using 11 networks trained on 3 paradigms, comprehensively demonstrate the effectiveness and generality of *Ctrl-GenAug*, particularly with pronounced performance gains in underrepresented high-risk populations and out-domain conditions.

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

