***This document provides examples to fine-tune Aria on three different datasets: single-image data, multi-image data, and video data.*** 📊

# Fine-tune on single-image dataset 🖼️
**1. Dataset Overview**  
We use a 30k subset of the [RefCOCO dataset](https://arxiv.org/pdf/1608.00272) as an example. RefCOCO is a visual grounding task where the model receives an image and a description of a reference object.  

**2. Task Description**  
The model is expected to output the corresponding bounding box. For a given bounding box, we normalize its coordinates to `[0,1000)` and transform it into "(x1,y1), (x2,y2)".  

**3. Additional Resources**  
Please refer to [RefCOCO_Example](./refcoco/README.md) for more details! 📄  


---

# Fine-tune on multi-image dataset 🖼️🖼️
**1. Dataset Overview**  
We use the [NLVR2 dataset](https://arxiv.org/abs/1811.00491) as an example.  

**2. Task Description**  
NLVR2 (Natural Language for Visual Reasoning) requires the model to determine whether a claim is true based on two images. The expected output is a simple yes or no answer.  

**3. Additional Resources**  
Please refer to [NLVR2_Example](./nlvr2/README.md) for details! ✅  

---

# Fine-tune on video dataset 🎥
**1. Dataset Overview**  
We use the [NextQA dataset](https://arxiv.org/abs/2105.08276) as an example.  

**2. Task Description**  
NextQA requires the model to select an answer from several options based on the video input and question posed. The model is expected to output the character corresponding to the correct option.  

**3. Additional Resources**  
Please refer to [NextQA_Example](./nextqa/README.md) for details! ✅  
