import math
from enum import Enum
from collections import namedtuple

import cv2
import torch
import numpy as np

from modules import devices, shared, prompt_parser, extra_networks
from modules.processing import opt_f

from tile_utils.typing import *


class ComparableEnum(Enum):

    def __eq__(self, other: Any) -> bool:
        if   isinstance(other, str):            return self.value == other
        elif isinstance(other, ComparableEnum): return self.value == other.value
        else: raise TypeError(f'unsupported type: {type(other)}')    

class Method(ComparableEnum):

    MULTI_DIFF = 'MultiDiffusion'
    MIX_DIFF   = 'Mixture of Diffusers'

class BlendMode(Enum):  # i.e. LayerType

    FOREGROUND = 'Foreground'
    BACKGROUND = 'Background'
    ADDITIONFOREGROUND = 'AdditionForeground'
    REMOVALFOREGROUND = 'RemovalForeground'
    EDITEDSOURCE = 'EditedSource'
    INVERSIONSOURCE = 'InversionSource'
    EDITEDTARGET = 'EditedTarget'
    EDITEDPAIR = 'EditedPair'
    WHOLE = 'Whole'

BBoxSettings = namedtuple('BBoxSettings', ['enable', 'x', 'y', 'w', 'h', 'prompt', 'neg_prompt', 'blend_mode', 'feather_ratio', 'seed', 'key_prompt'])
NoiseInverseCache = namedtuple('NoiseInversionCache', ['model_hash', 'x0', 'xt', 'noise_inversion_steps', 'retouch', 'prompts'])
DEFAULT_BBOX_SETTINGS = BBoxSettings(False, 0.4, 0.4, 0.2, 0.2, '', '', BlendMode.BACKGROUND.value, 0.2, -1, '')
NUM_BBOX_PARAMS = len(BBoxSettings._fields)


def build_bbox_settings(bbox_control_states:List[Any]) -> Dict[int, BBoxSettings]:
    settings = {}
    for index, i in enumerate(range(0, len(bbox_control_states), NUM_BBOX_PARAMS)):
        setting = BBoxSettings(*bbox_control_states[i:i+NUM_BBOX_PARAMS])
        # for float x, y, w, h, feather_ratio, keeps 4 digits
        setting = setting._replace(
            x=round(setting.x, 4), 
            y=round(setting.y, 4), 
            w=round(setting.w, 4), 
            h=round(setting.h, 4), 
            feather_ratio=round(setting.feather_ratio, 4),
            seed=int(setting.seed),
        )
        # sanity check
        if not setting.enable or setting.x > 1.0 or setting.y > 1.0 or setting.w <= 0.0 or setting.h <= 0.0: continue
        settings[index] = setting
    return settings

def gr_value(value=None, visible=None):
    return {"value": value, "visible": visible, "__type__": "update"}


class BBox:

    ''' grid bbox '''

    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.box = [x, y, x+w, y+h]
        self.slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)

    def __getitem__(self, idx:int) -> int:
        return self.box[idx]

class CustomBBox(BBox):

    ''' region control bbox '''

    def __init__(self, x:int, y:int, w:int, h:int, prompt:str, neg_prompt:str, blend_mode:str, feather_radio:float, seed:int, key_prompt:str):
        super().__init__(x, y, w, h)
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.key_prompt = key_prompt
        self.blend_mode = BlendMode(blend_mode)
        self.feather_ratio = max(min(feather_radio, 1.0), 0.0)
        self.seed = seed
        # initialize necessary fields
        self.feather_mask = feather_mask(self.w, self.h, self.feather_ratio) if self.blend_mode != BlendMode.BACKGROUND else None
        self.cond: MulticondLearnedConditioning = None
        self.extra_network_data: DefaultDict[List[ExtraNetworkParams]] = None
        self.uncond: List[List[ScheduledPromptConditioning]] = None
        self.batch_prompts:List[str] = None
        self.batch_neg_prompts:List[str] = None

class EditBBox(BBox):

    ''' region control bbox '''

    def __init__(self, x:int, y:int, w:int, h:int, paired_prompt:dict,  blend_mode:str, feather_radio:float, seed:int):
        super().__init__(x, y, w, h)
        #paired_prompt['src'] = [p,np,kp] 
        self.prompt = [paired_prompt['src'][0], paired_prompt['target'][0]]
        self.neg_prompt = [paired_prompt['src'][1], paired_prompt['target'][1]]
        self.key_prompt = [paired_prompt['src'][2], paired_prompt['target'][2]]
        self.blend_mode = BlendMode(blend_mode)
        self.feather_ratio = max(min(feather_radio, 1.0), 0.0)
        self.seed = seed
        # initialize necessary fields
        self.feather_mask = feather_mask(self.w, self.h, self.feather_ratio) if self.blend_mode != BlendMode.BACKGROUND else None
        self.cond: MulticondLearnedConditioning = None
        self.extra_network_data: DefaultDict[List[ExtraNetworkParams]] = None
        self.uncond: List[List[ScheduledPromptConditioning]] = None
        self.batch_prompts:List[str] = None
        self.batch_neg_prompts:List[str] = None

class PromptToPromptBBox(BBox):

    ''' region control bbox '''

    def __init__(self, x:int, y:int, w:int, h:int, prompt:List[str], neg_prompt:List[str], region_ids:List[List[int]], neg_region_ids:List[List[int]], blend_mode:str, feather_radio:float, seed:int, key_prompt:str, key_region_ids:List[List[int]]):
        super().__init__(x, y, w, h)
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.key_prompt = key_prompt
        self.region_ids = region_ids
        self.neg_region_ids = neg_region_ids
        self.key_region_ids = key_region_ids
        self.blend_mode = BlendMode(blend_mode)
        self.feather_ratio = max(min(feather_radio, 1.0), 0.0)
        self.seed = seed
        # initialize necessary fields
        self.feather_mask = feather_mask(self.w, self.h, self.feather_ratio) if self.blend_mode == BlendMode.FOREGROUND or self.blend_mode == BlendMode.ADDITIONFOREGROUND or self.blend_mode == BlendMode.WHOLE else None
        self.cond: MulticondLearnedConditioning = None
        self.extra_network_data: DefaultDict[List[ExtraNetworkParams]] = None
        self.uncond: List[List[ScheduledPromptConditioning]] = None
        self.batch_prompts:List[str] = None
        self.batch_neg_prompts:List[str] = None

        words = [cur_prompt.split(',') for cur_prompt in self.key_prompt]
        longer_id = 0 if len(self.key_region_ids[0]) > len(self.key_region_ids[1]) else 1
        shorter_id = 1 - longer_id

        self.region_key_words = {longer_id:{}, shorter_id:{}}
        for ri, region_id in enumerate(self.key_region_ids[longer_id]):
            if region_id not in self.region_key_words[longer_id].keys(): self.region_key_words[longer_id][region_id] = []
            self.region_key_words[longer_id][region_id].append(words[longer_id][ri])
        
        for ri, region_id in enumerate(self.key_region_ids[shorter_id]):
            if region_id not in self.region_key_words[shorter_id].keys(): self.region_key_words[shorter_id][region_id] = []
            self.region_key_words[shorter_id][region_id].append(words[shorter_id][ri])
                    


class Prompt:

    ''' prompts helper '''

    @staticmethod
    def apply_styles(prompts:List[str], styles=None) -> List[str]:
        if not styles: return prompts
        return [shared.prompt_styles.apply_styles_to_prompt(p, styles) for p in prompts]

    @staticmethod
    def append_prompt(prompts:List[str], prompt:str='') -> List[str]:
        if not prompt: return prompts
        return [f'{p}, {prompt}' for p in prompts]
    
    @staticmethod
    def append_edited_prompt(prompts:List[str], edit_prompt:List[str]) -> List[str]:
        if not edit_prompt: return prompts
        res= []
        for global_prompt, region_prompt in zip(prompts, edit_prompt):
            tmp_region = f', {region_prompt}' if region_prompt != '' else f''
            res.append(f'{global_prompt}{tmp_region}')
        return res

class Condition:

    ''' CLIP cond helper '''

    @staticmethod
    def get_custom_cond(prompts:List[str], prompt, steps:int, styles=None, blend_mode=None) -> Tuple[Cond, ExtraNetworkData]:
        if blend_mode == BlendMode.WHOLE or blend_mode == BlendMode.EDITEDPAIR:
            if not isinstance(prompt,list):
                raise ValueError(f"in the prompt-to-prompt editing mode, the prompts in BlendMode.WHOLE should be a list with the same number of elements with batchsize.")
            prompt = Prompt.apply_styles(prompt, styles)
            _, extra_network_data = extra_networks.parse_prompts(prompt)
            prompts = Prompt.append_edited_prompt(prompts, prompt)
        else:
            prompt = Prompt.apply_styles([prompt], styles)[0]
            _, extra_network_data = extra_networks.parse_prompts([prompt])
            #修改：在此处添加对ADDITIONFOREGROUND 和 FOREGROUND的区别处理，不同的batch进行不同的赋值
            #  FOREGROUND = 'Foreground'
            # BACKGROUND = 'Background'
            # ADDITIONFOREGROUND = 'AdditionForeground'
            # REMOVALFOREGROUND = 'RemovalForeground'
            # EDITEDSOURCE = 'EditedSource'
            # INVERSIONSOURCE = 'InversionSource'
            # EDITEDTARGET = 'EditedTarget'
            # EDITEDPAIR = 'EditedPair'
            if blend_mode == BlendMode.ADDITIONFOREGROUND:
                prompts = Prompt.append_edited_prompt(prompts, ['', prompt])
            elif blend_mode == BlendMode.REMOVALFOREGROUND:
                prompts = Prompt.append_edited_prompt(prompts, [prompt, ''])
            # elif blend_mode == BlendMode.EDITEDPAIR:
            #     prompts = Prompt.append_edited_prompt(prompts, prompt)
            elif blend_mode == BlendMode.FOREGROUND or blend_mode == BlendMode.BACKGROUND:
                prompts = Prompt.append_prompt(prompts, prompt)
        prompts = Prompt.apply_styles(prompts, styles)
        cond = Condition.get_cond(prompts, steps)
        return cond, extra_network_data, prompts
    
    @staticmethod
    def get_cond(prompts, steps:int):
        prompts, _ = extra_networks.parse_prompts(prompts)
        cond = prompt_parser.get_multicond_learned_conditioning(shared.sd_model, prompts, steps) #For each prompt, the list is obtained by splitting the prompt using the AND separator
        return cond

    @staticmethod
    def get_uncond(neg_prompts:List[str], steps:int, styles=None) -> Uncond:
        neg_prompts = Prompt.apply_styles(neg_prompts, styles)
        uncond = prompt_parser.get_learned_conditioning(shared.sd_model, neg_prompts, steps)
        return uncond

    @staticmethod
    def get_uncond_editversion(neg_prompts:List[str], prompt, steps:int, styles=None, blend_mode=None) -> Uncond:

        if blend_mode == BlendMode.ADDITIONFOREGROUND:
            neg_prompts = Prompt.append_edited_prompt(neg_prompts, ['', prompt])
        elif blend_mode == BlendMode.REMOVALFOREGROUND:
            neg_prompts = Prompt.append_edited_prompt(neg_prompts, [prompt, ''])
        elif blend_mode == BlendMode.EDITEDPAIR or blend_mode == BlendMode.WHOLE:
            neg_prompts = Prompt.append_edited_prompt(neg_prompts, prompt)
        elif blend_mode == BlendMode.FOREGROUND or blend_mode == BlendMode.BACKGROUND:
            neg_prompts = Prompt.append_prompt(neg_prompts, prompt)
            
        neg_prompts = Prompt.apply_styles(neg_prompts, styles)
        uncond = prompt_parser.get_learned_conditioning(shared.sd_model, neg_prompts, steps)
        return uncond, neg_prompts

    @staticmethod
    def reconstruct_cond(cond:Cond, step:int) -> Tensor:
        list_of_what, tensor = prompt_parser.reconstruct_multicond_batch(cond, step)
        return tensor

    def reconstruct_uncond(uncond:Uncond, step:int) -> Tensor:
        tensor = prompt_parser.reconstruct_cond_batch(uncond, step)
        return tensor


def splitable(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16) -> bool:
    w, h = w // opt_f, h // opt_f
    min_tile_size = min(tile_w, tile_h)
    if overlap >= min_tile_size:
        overlap = min_tile_size - 4
    cols = math.ceil((w - overlap) / (tile_w - overlap))
    rows = math.ceil((h - overlap) / (tile_h - overlap))
    return cols > 1 or rows > 1

def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = math.ceil((w - overlap) / (tile_w - overlap))
    rows = math.ceil((h - overlap) / (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight


def gaussian_weights(tile_w:int, tile_h:int) -> Tensor:
    '''
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.
    '''
    from numpy import pi, exp, sqrt
    
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]   # -1 because index goes from 0 to latent_width - 1
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)

def feather_mask(w:int, h:int, ratio:float) -> Tensor:
    '''Generate a feather mask for the bbox'''

    mask = np.ones((h, w), dtype=np.float32)
    feather_radius = int(min(w//2, h//2) * ratio)
    # Generate the mask via gaussian weights
    # adjust the weight near the edge. the closer to the edge, the lower the weight
    # weight = ( dist / feather_radius) ** 2
    for i in range(h//2):
        for j in range(w//2):
            dist = min(i, j)
            if dist >= feather_radius: continue
            weight = (dist / feather_radius) ** 2
            mask[i, j] = weight
            mask[i, w-j-1] = weight
            mask[h-i-1, j] = weight
            mask[h-i-1, w-j-1] = weight

    return torch.from_numpy(mask).to(devices.device, dtype=torch.float32)

def get_retouch_mask(img_input: np.ndarray, kernel_size: int) -> np.ndarray:
    '''
    Return the area where the image is retouched.
    Copy from Zhihu.com
    '''
    step   = 1
    kernel = (kernel_size, kernel_size)
    
    img    = img_input.astype(np.float32)/255.0
    sz     = img.shape[:2]
    sz1    = (int(round(sz[1] * step)), int(round(sz[0] * step)))
    sz2    = (int(round(kernel[0] * step)), int(round(kernel[0] * step)))
    sI     = cv2.resize(img, sz1, interpolation=cv2.INTER_LINEAR)
    sp     = cv2.resize(img, sz1, interpolation=cv2.INTER_LINEAR)
    msI    = cv2.blur(sI, sz2)
    msp    = cv2.blur(sp, sz2)
    msII   = cv2.blur(sI*sI, sz2)
    msIp   = cv2.blur(sI*sp, sz2)
    vsI    = msII - msI*msI
    csIp   = msIp - msI*msp
    recA   = csIp/(vsI+0.01)
    recB   = msp - recA*msI
    mA     = cv2.resize(recA, (sz[1],sz[0]), interpolation=cv2.INTER_LINEAR)
    mB     = cv2.resize(recB, (sz[1],sz[0]), interpolation=cv2.INTER_LINEAR)
    
    gf = mA * img + mB
    gf -= img
    gf *= 255
    gf = gf.astype(np.uint8)
    gf = gf.clip(0, 255)
    gf = gf.astype(np.float32)/255.0
    return gf


def null_decorator(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return wrapper

keep_signature = null_decorator
controlnet     = null_decorator
stablesr       = null_decorator
grid_bbox      = null_decorator
custom_bbox    = null_decorator
noise_inverse  = null_decorator
