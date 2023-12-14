import torch
from torch import nn, einsum
from einops import rearrange, repeat
import os
from inspect import isfunction
from typing import Optional, Union, Tuple, List, Callable, Dict
from tile_methods import ptp_utils, seq_aligner
import torch.nn.functional as nnf
import numpy as np
import abc
from modules import prompt_parser
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
LOW_RESOURCE = False 
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77

class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step] #在采样过程的不同t内，用不同的alpha的值去控制两者attn的加权之和
                # get_avg_cross_attn(attn_base, attn_repalce[0])
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 tokenizer = None, device = None):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, tokenizer = None, device: Optional[torch.device] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer = tokenizer, device = device)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha_linear(prompts, num_steps, cross_replace_steps, tokenizer, max_num_words = self.mapper.shape[1]).to(device)

        
class AttentionManipulate(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, mask_num_word = False, tokenizer = None, device: Optional[torch.device] = None):
        super(AttentionManipulate, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer = tokenizer, device = device)
        self.mapper, alphas = seq_aligner.get_manipulation_mapper(prompts, tokenizer, mask_num_word = mask_num_word)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1]) # 1,77 --> 1,1,1,77
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha_linear(prompts, num_steps, cross_replace_steps, tokenizer, max_num_words = self.mapper.shape[1]).to(device)

class MultiRegionAttentionController():
    def __init__(self , tokenizer = None, device: Optional[torch.device] = None, select_self = True):
        self.conrtollers = {}
        self.region_key = ''
        self.tokenizer = tokenizer
        self.device = device
        self.key_cross_attn_index = None
        self.key_self_attn_index = {}
        self.whole_id = -1
        self.select_self = select_self
        self.self_attn_scale = 1.5
        self.key_self_attn_layer = ['down','up']
    
    def select_region(self, region_key):
        self.region_key = region_key

    def show_cross_attn(self,region_key, prompts:List[str]):
        attns0, words0 = show_cross_attention_regional(self.conrtollers[region_key], res=16,prompts = prompts, from_where=("up", "down"), select = 0, tokenizer = self.tokenizer)
        attns1, words1 = show_cross_attention_regional(self.conrtollers[region_key], res=16,prompts = prompts, from_where=("up", "down"), select = 1, tokenizer = self.tokenizer)
        return [attns0, attns1], [words0, words1]
    
    def cal_cross_attn(self,region_key, prompts:List[str]):
        attns0, words0, indexs0 = cal_cross_attention_for_key_word(self.conrtollers[region_key], res=16,prompts = prompts, from_where=("up", "down"), select = 0, tokenizer = self.tokenizer)
        attns1, words1, indexs1 = cal_cross_attention_for_key_word(self.conrtollers[region_key], res=16,prompts = prompts, from_where=("up", "down"), select = 1, tokenizer = self.tokenizer)
        return [attns0, attns1 ], [words0, words1], [indexs0, indexs1]

    def make_manipulate(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, mask_num_word = False):
        if self.region_key not in self.conrtollers.keys():
            self.conrtollers[self.region_key] = AttentionManipulate(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer = self.tokenizer, device = self.device)

    def make_replace(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, mask_num_word = False):
        if self.region_key not in self.conrtollers.keys():
            self.conrtollers[self.region_key] = AttentionReplace(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer = self.tokenizer, device = self.device)

    def make_refine(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None, mask_num_word = False):
        if self.region_key not in self.conrtollers.keys():
            self.conrtollers[self.region_key] = AttentionRefine(prompts, num_steps, cross_replace_steps, self_replace_steps, tokenizer = self.tokenizer, device = self.device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer = tokenizer, device = device)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend, tokenizer = tokenizer, device = device)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer


from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def aggregate_attention_regional(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int, prompts:str):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()

def show_cross_attention_regional(attention_store: AttentionStore, res: int, from_where: List[str], prompts:List[str], select: int = 0, tokenizer = None):
    #tokens = tokenizer.encode(prompts[select])

    parsed_x = prompt_parser.parse_prompt_attention(prompts[select])

    #tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
    x_seq =  []
    for text, _ in parsed_x:
        x_seq.append(tokenizer.encode(text, truncation=False, add_special_tokens=False))

    # x_seq = [tokenizer.encode(text) for text, _ in parsed_x]
    # y_seq = [tokenizer.encode(text) for text, _ in parsed_y]
    tokens, total_nums = seq_aligner.get_web_ui_chunk_prompt(x_seq)

    decoder = tokenizer.decode
    attention_maps = aggregate_attention_regional(attention_store, res, from_where, True, select, prompts)
    images = []
    words = []
    for i in range(len(tokens)):
        word = decoder(int(tokens[i]))
        if word == ',': continue
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, word)
        images.append(image)
        words.append(word)
    attn_img = view_images(np.stack(images, axis=0))
    return attn_img, words

def cal_cross_attention_for_key_word(attention_store: AttentionStore, res: int, from_where: List[str], prompts:List[str], select: int = 0, tokenizer = None):
    #tokens = tokenizer.encode(prompts[select])

    parsed_x = prompt_parser.parse_prompt_attention(prompts[select])

    #tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]
    x_seq =  []
    for text, _ in parsed_x:
        x_seq.append(tokenizer.encode(text, truncation=False, add_special_tokens=False))

    # x_seq = [tokenizer.encode(text) for text, _ in parsed_x]
    # y_seq = [tokenizer.encode(text) for text, _ in parsed_y]
    tokens, total_nums = seq_aligner.get_web_ui_chunk_prompt(x_seq)

    decoder = tokenizer.decode
    attention_maps = aggregate_attention_regional(attention_store, res, from_where, True, select, prompts)
    images = []
    words = []
    indexs = []
    for i in range(len(tokens)):
        word = decoder(int(tokens[i]))
        if word == ',': continue
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 1)  #   expand.3
        image = image.numpy().astype(np.uint8)
        # image = np.array(Image.fromarray(image).resize((256, 256)))
        # image = ptp_utils.text_under_image(image, word)
        images.append(image)
        words.append(word)
        indexs.append(i)
    # attn_img = view_images(np.stack(images, axis=0))
    return images, words, indexs

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    # pil_img = Image.fromarray(image_)
    return image_


def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1))



def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out


        def forward(x, context=None, mask=None):
            h = self.heads

            q = self.to_q(x)
            is_cross = context is not None
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            # force cast to fp32 to avoid overflowing
            if _ATTN_PRECISION =="fp32":
                with torch.autocast(enabled=False, device_type = 'cuda'):
                    q, k = q.float(), k.float()
                    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            else:
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            
            del q, k
        
            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)

            # use the guidance from total average cross-attn, we reweight the key word
            if controller.select_self and (controller.region_key == controller.whole_id) and controller.key_cross_attn_index != None:
                if place_in_unet in controller.key_self_attn_layer:
                    if not is_cross:
                        if place_in_unet in controller.key_self_attn_index.keys():
                            if sim.shape[-1] in controller.key_self_attn_index[place_in_unet]:
                                key_self_index = controller.key_self_attn_index[place_in_unet][sim.shape[-1]]
                                sim[:,:,key_self_index] *= controller.self_attn_scale
                    else:
                        key_self_index = controller.key_cross_attn_index
                        key_sim = sim[:,:,key_self_index].detach()
                        key_sim_sum = key_sim.sum(dim=-1).sum(dim=0)
                        _, self_index = torch.topk(key_sim_sum,key_sim_sum.shape[0]//2)

                        if place_in_unet not in controller.key_self_attn_index.keys():
                            controller.key_self_attn_index[place_in_unet] = {}
                        controller.key_self_attn_index[place_in_unet][key_sim_sum.shape[0]] = self_index.detach()


            
            # attention, what we cannot get enough of
            sim = sim.softmax(dim=-1)
            # if controller.region_key == 'Whole':
            sim = controller.conrtollers[controller.region_key](sim, is_cross, place_in_unet)
            out = einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        return forward
    
    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    for c in controller.conrtollers.keys():
        if controller.conrtollers[c] is None:
            controller.conrtollers[c] = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "input" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "output" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "middle" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    for c in controller.conrtollers.keys():

        controller.conrtollers[c].num_att_layers = cross_att_count




        # def forward(x, encoder_hidden_states=None, attention_mask=None):
        #     batch_size, sequence_length, dim = x.shape
        #     h = self.heads
        #     q = self.to_q(x)
        #     is_cross = encoder_hidden_states is not None
        #     encoder_hidden_states = encoder_hidden_states if is_cross else x
        #     k = self.to_k(encoder_hidden_states)
        #     v = self.to_v(encoder_hidden_states)
        #     q = self.head_to_batch_dim(q)
        #     k = self.head_to_batch_dim(k)
        #     v = self.head_to_batch_dim(v)

        #     sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

        #     if attention_mask is not None:
        #         attention_mask = attention_mask.reshape(batch_size, -1)
        #         max_neg_value = -torch.finfo(sim.dtype).max
        #         attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
        #         sim.masked_fill_(~attention_mask, max_neg_value)

        #     # attention, what we cannot get enough of
        #     attn = sim.softmax(dim=-1)
        #     # if is_cross and batch_size > 2: get_avg_cross_attn(sim, batch_size//2, place_in_unet)
        #     attn = controller(attn, is_cross, place_in_unet)
        #     out = torch.einsum("b i j, b j d -> b i d", attn, v)
        #     out = self.batch_to_head_dim(out)
        #     return to_out(out)

        # return forward
