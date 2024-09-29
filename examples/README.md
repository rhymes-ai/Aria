***This document provides examples to fine-tune Aria on three different datasets: single-image data, multi-image data and video data.***

# Single-Image SFT
We use a 30k subset of the [RefCOCO dataset](https://arxiv.org/pdf/1608.00272) as an example.
RefCOCO is a visual grounding task. Given an image and a description of the reference object as input, the model is expected to output corresponding bounding box. For a given bounding box, we normalize its coordinates to `[0,1000)` and transform it into "(x1,y1), (x2,y2)". Please refer to [RefCOCO_Example](./refcoco/README.md) for more details!



# Multi-Image SFT
We use the [NLVR2 dataset](https://arxiv.org/abs/1811.00491) as an example. 
NLVR2 (Natural Language for Visual Reasoning) is a task where given two images, the model needs to determine whether a claim is true by answering yes or no. Please refer to [NLVR2_Example](./nlvr2/README.md) for details!


# Video SFT
We use the [NextQA dataset](https://arxiv.org/abs/2105.08276) as an example.
NextQA requires the model to select an answer from several options according to the video input and question. The model is expected to output the correct option's character. Please refer to [NextQA_Example](./nextqa/README.md) for details!

