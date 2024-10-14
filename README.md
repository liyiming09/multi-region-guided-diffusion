# multi-region-guided-diffusion for sd-webui

A plug-in for stable diffusion webUI. Users can perform region/object level image manipulation, including object addition, removal,  attribute modification and relation manipulation.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This extension is licensed under [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), everyone is FREE of charge to access, use, modify and redistribute with the same license.  
**You cannot use versions after AOE 2023.3.28 for commercial sales (only refers to code of this repo, the derived artworks are NOT restricted).**

****


The extension helps you to **manipulate images (512 * 512) including object addition, removal and relation manipulatiopn** via the following techniques:

- Reproduced SOTA Tiled Diffusion methods
  - [MultiDiffusion](https://multidiffusion.github.io)
  - [Tiled Diffusion & VAE extension for sd-webui](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
- Our original MRGD method for generation
- My original MRGD Inversion method for real-world image inversion


### Features

- Core
  - [x] [Tiled Diffusion](#tiled-Diffusion)
  - [x] [Regional Prompt Control based on MRGD](#region-prompt-control)
  - [x] [Tiled Noise Inversion](#tiled-noise-inversion)

### Installation
- install from git URL
  - Open Automatic1111 WebUI -> Click Tab "Extensions" -> Click Tab "Install from URL" -> type in [https://github.com/liyiming09/multi-region-guided-diffusion.git](https://github.com/liyiming09/multi-region-guided-diffusion) -> Click "Install"


### Examples of Regional Prompt Control
In the image manipulation task, we only consider the case of multi-object generation . Therefore, we do not consider the case of large-scale image generation in the Tiled Diffusion plug-in code.

- Basic settings：
  - model: realisticVision_v51, 512 * 512
  - sampling method = DDIM, sampling steps = 20, Batch size = 2
  - Enable Tiled Diffusion = True, method=MultiDiffusion, Enable Region Prompt Control = True

#### Example 1: Relation manipulation
- Prompts:
  - Main prompt = medium shot, cinematic shot, cinematic lighting, best quality, HDR, raw photo, film grain, cinestill 50d
  - Negative prompt =  (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), ...
  - Region 1: Prompt = outdoors, park, **Type = Background**
  - Region 2: Prompt = 1girl, (masterpiece), (extremely intricate:1.3),white blue dress, (realistic), the most beautiful in the world, upper body, outdoors, intense sunlight, far away,sharp focus, dramatic, award winning, cinematic lighting, octane render, unreal engine,(intrinsically detailed face:1.3), stand with another girl, **Type = Foreground**, Feather = 0.3
  - Region 3: Prompt = a young brunette stand with another girl, long messy hair, warm turtleneck, (full body:1.8), sharp focus, (red dress:1.3),(intrinsically detailed face:1.3), outdoors, far away, **Type = AddRelation**, Feather = 0.2
  - Region Layout:
    ![alt text](https://github.com/liyiming09/RMD-img-demo/blob/main/MRGD/image.png?raw=true)
- Relation settings:
  ![alt text](https://github.com/liyiming09/RMD-img-demo/blob/main/MRGD/image-2.png?raw=true)
  - with pre-trained lora weights and a word-embedding
- Results:
  ![alt text](https://github.com/liyiming09/RMD-img-demo/blob/main/MRGD/1-0.png?raw=true)

#### Example 2: Object addition
- Prompts:
  - Main prompt = medium shot, cinematic shot, cinematic lighting, best quality, HDR, raw photo, film grain, cinestill 50d
  - Negative prompt =  (deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), ...
  - Region 1: Prompt = outdoors, castle, (Disney Castle) ,far away castle, **Type = Background**
  - Region 2: Prompt = 1girl, (princess snow white,Disney princess), (masterpiece), (extremely intricate:1.3),white blue dress, (realistic), the most beautiful in the world, upper body, outdoors, intense sunlight, far away castle,sharp focus, dramatic, award winning, cinematic lighting, octane render, unreal engine,(intrinsically detailed face:1.3), **Type = Foreground**, Feather = 0.3
  - Region 3: Prompt = a young brunette, long messy hair, warm turtleneck, (full body:1.8), sharp focus, (red dress:1.3),(intrinsically detailed face:1.3), holding hands with the person around,outdoors, far away castle, **Type = AdditionForeground**, Feather = 0.2
- Relation settings:
  - None
- Results:
  ![alt text](https://github.com/liyiming09/RMD-img-demo/blob/main/MRGD/add-10-real.png?raw=true)

### Benchmark
- Most of the prompts about the synthesized object come from publicly available samples on the open-source website civitai.org, some of which have been slightly modified.
- Reproducibility:
  - A regional config file is provided for each instance, as well as a generated log file containing the details of each prompt and the random seed.
  - Due to the optimization of the algorithm, the hyperparameters are set differently in different versions of the algorithm, so you need to pay attention to the specific settings of the hyperparameters in the tiled_diffusion.py file. The key hyperparameters will be integrated into the UI interface later on
### Others
- We would like to thank the following repos, their code was essential in the developement of this project: [https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111](https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111)
- This code is a plug-in developed for [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
Because in the subsequent development process, WebUI official sampler (especially DDIM Sampler) and some interface code changes, the current version of the code can not run on the latest version of WebUI, will be updated after completing the subsequent adjustments.
### Citation
  If you find our repo useful for your research, please consider citing our paper:
  ```bibtex
  @inproceedings{li2024multi,
  title={Multi-Region Text-Driven Manipulation of Diffusion Imagery},
  author={Li, Yiming and Zhou, Peng and Sun, Jun and Xu, Yi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={4},
  pages={3261--3269},
  year={2024}
}
  ```

### License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


