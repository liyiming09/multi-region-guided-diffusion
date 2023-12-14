from os import curdir
import torch

from modules import devices, extra_networks
from modules.shared import state

from tile_methods.abstractdiffusion import TiledDiffusion
from tile_utils.utils import *
from tile_utils.typing import *


class MultiDiffusion(TiledDiffusion):
    """
        Multi-Diffusion Implementation
        https://arxiv.org/abs/2302.08113
    """

    def __init__(self, p:Processing, *args, **kwargs):
        super().__init__(p, *args, **kwargs)
        assert p.sampler_name != 'UniPC', 'MultiDiffusion is not compatible with UniPC!'

        # For ddim sampler we need to cache the pred_x0
        self.x_pred_buffer = None

    def hook(self):
        if self.is_kdiff:
            # For K-Diffusion sampler with uniform prompt, we hijack into the inner model for simplicity
            # Otherwise, the masked-redraw will break due to the init_latent
            self.sampler: CFGDenoiser
            self.sampler_forward = self.sampler.inner_model.forward
            self.sampler.inner_model.forward = self.kdiff_forward
        else:
            self.sampler: VanillaStableDiffusionSampler
            self.sampler_forward = self.sampler.orig_p_sample_ddim
            self.sampler.orig_p_sample_ddim = self.ddim_forward

    @staticmethod
    def unhook():
        # no need to unhook MultiDiffusion as it only hook the sampler,
        # which will be destroyed after the painting is done
        pass

    def reset_buffer(self, x_in:Tensor):
        super().reset_buffer(x_in)
        
        # ddim needs to cache pred0
        if self.is_ddim:
            if self.x_pred_buffer is None:
                self.x_pred_buffer = torch.zeros_like(x_in, device=x_in.device)
            else:
                self.x_pred_buffer.zero_()

    @custom_bbox
    def init_custom_bbox(self, *args):
        super().init_custom_bbox(*args)

        for bbox in self.custom_bboxes:
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.weights[bbox.slicer] += 1.0

    ''' ↓↓↓ kernel hijacks ↓↓↓ '''

    def repeat_cond_dict(self, cond_input:CondDict, bboxes:List[CustomBBox]) -> CondDict:
        cond = cond_input['c_crossattn'][0]
        # repeat the condition on its first dim
        cond_shape = cond.shape
        cond = cond.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
        image_cond = self.get_image_cond(cond_input)
        if image_cond.shape[2] == self.h and image_cond.shape[3] == self.w:
            image_cond_list = []
            for bbox in bboxes:
                image_cond_list.append(image_cond[bbox.slicer])
            image_cond_tile = torch.cat(image_cond_list, dim=0)
        else:
            image_cond_shape = image_cond.shape
            image_cond_tile = image_cond.repeat((len(bboxes),) + (1,) * (len(image_cond_shape) - 1))
        return self.make_condition_dict([cond], image_cond_tile)

    @torch.no_grad()
    @keep_signature
    def kdiff_forward(self, x_in:Tensor, sigma_in:Tensor, cond:CondDict) -> Tensor:
        '''
        This function hijacks `k_diffusion.external.CompVisDenoiser.forward()`
        So its signature should be the same as the original function, especially the "cond" should be with exactly the same name
        '''

        assert CompVisDenoiser.forward

        def org_func(x:Tensor):
            return self.sampler_forward(x, sigma_in, cond=cond)

        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]):
            # For kdiff sampler, the dim 0 of input x_in is:
            #   = batch_size * (num_AND + 1)   if not an edit model
            #   = batch_size * (num_AND + 2)   otherwise
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(cond, bboxes)
            x_tile_out = self.sampler_forward(x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out

        def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox):
            return self.kdiff_custom_forward(x, sigma_in, cond, bbox_id, bbox, self.sampler_forward)

        return self.sample_one_step(x_in, org_func, repeat_func, custom_func) #sd_samplers_kdiffusion.py line.126

    @torch.no_grad()
    @keep_signature
    def ddim_forward(self, x_in:Tensor, cond_in:Union[CondDict, Tensor], ts:Tensor, unconditional_conditioning:Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        '''
        This function will replace the original p_sample_ddim function in ldm/diffusionmodels/ddim.py
        So its signature should be the same as the original function,
        Particularly, the unconditional_conditioning should be with exactly the same name
        '''

        assert VanillaStableDiffusionSampler.p_sample_ddim_hook

        def org_func(x:Tensor):
            return self.sampler_forward(x, cond_in, ts, unconditional_conditioning=unconditional_conditioning, *args, **kwargs)

        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]):
            if isinstance(cond_in, dict):
                ts_tile    = ts.repeat(len(bboxes))
                cond_tile  = self.repeat_cond_dict(cond_in, bboxes)
                ucond_tile = self.repeat_cond_dict(unconditional_conditioning, bboxes)
            else:
                ts_tile = ts.repeat(len(bboxes))
                cond_shape  = cond_in.shape
                cond_tile   = cond_in.repeat((len(bboxes),) + (1,) * (len(cond_shape) - 1))
                ucond_shape = unconditional_conditioning.shape
                ucond_tile  = unconditional_conditioning.repeat((len(bboxes),) + (1,) * (len(ucond_shape) - 1))
            x_tile_out, x_pred = self.sampler_forward(
                x_tile, cond_tile, ts_tile, 
                unconditional_conditioning=ucond_tile, 
                *args, **kwargs)
            return x_tile_out, x_pred

        def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox):
            # before the final forward, we can set the control tensor
            def forward_func(x, *args, **kwargs):
                self.set_custom_controlnet_tensors(bbox_id, 2*x.shape[0])
                self.set_custom_stablesr_tensors(bbox_id)
                return self.sampler_forward(x, *args, **kwargs)
            return self.ddim_custom_forward(x, cond_in, bbox, ts, forward_func, bbox_id = bbox_id, *args, **kwargs)

        return self.sample_one_step(x_in, org_func, repeat_func, custom_func, *args, **kwargs)

    def sample_one_step(self, x_in:Tensor, org_func: Callable, repeat_func:Callable, custom_func:Callable, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        '''
        this method splits the whole latent and process in tiles
            - x_in: current whole U-Net latent
            - org_func: original forward function, when use highres
            - denoise_func: one step denoiser for grid tile
            - denoise_custom_func: one step denoiser for custom tile
        '''
        def set_requires_grad(model, value):
            for param in model.parameters():
                param.requires_grad = value

        N, C, H, W = x_in.shape
        if H != self.h or W != self.w:
            self.reset_controlnet_tensors()
            return org_func(x_in)

        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted: return x_in

                # batching
                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                x_tile = torch.cat(x_tile_list, dim=0)

                # controlnet tiling
                # FIXME: is_denoise is default to False, however it is set to True in case of MixtureOfDiffusers
                self.switch_controlnet_tensors(batch_id, N, len(bboxes))

                # stablesr tiling
                self.switch_stablesr_tensors(batch_id)


                self.controller.select_region('first')
                # compute tiles
                if self.is_kdiff:
                    x_tile_out = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                else:
                    x_tile_out, x_tile_pred = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer     [bbox.slicer] += x_tile_out [i*N:(i+1)*N, :, :, :]
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred[i*N:(i+1)*N, :, :, :]

                # update progress bar
                self.update_pbar()

        # Custom region sampling (custom bbox)
        x_feather_buffer      = None
        x_feather_mask        = None
        x_feather_count       = None
        x_feather_pred_buffer = None
        if len(self.custom_bboxes) > 0:
            for bbox_id, bbox in enumerate(self.custom_bboxes):
                #在分区域的region latent noise 具有足够的引导时，才启用WHOLE的交互引导
                if kwargs['index'] > self.guide_step[3] * self.p.steps: 
                    if  bbox.blend_mode == BlendMode.WHOLE: 
                        continue
                # else:
                #     if bbox.blend_mode == BlendMode.ADDITIONFOREGROUND:
                #         continue
                if state.interrupted: return x_in

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.activate(self.p, bbox.extra_network_data)

                x_tile = x_in[bbox.slicer]

                self.controller.select_region(bbox_id)
                if self.is_kdiff:
                    # retrieve original x_in from construncted input
                    x_tile_out = custom_func(x_tile, bbox_id, bbox)

                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer[bbox.slicer] += x_tile_out
                    else:
                    # elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer = torch.zeros_like(self.x_buffer)
                            x_feather_mask   = torch.zeros((1, 1, H, W), device=x_in.device)
                            x_feather_count  = torch.zeros((1, 1, H, W), device=x_in.device)
                        x_feather_buffer[bbox.slicer] += x_tile_out
                        x_feather_mask  [bbox.slicer] += bbox.feather_mask
                        x_feather_count [bbox.slicer] += 1
                else:
                    #here, in DDIM, apply the original DDIM p_sample, otherwise apply the sample function in K-diff way
                    x_tile_out, x_tile_pred = custom_func(x_tile, bbox_id, bbox)


                    ######################################################33
                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer     [bbox.slicer] += x_tile_out
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred
                    else:
                    # elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer      = torch.zeros_like(self.x_buffer)
                            x_feather_pred_buffer = torch.zeros_like(self.x_pred_buffer)
                            x_feather_mask        = torch.zeros((2, 1, H, W), device=x_in.device)
                            x_feather_count       = torch.zeros((2, 1, H, W), device=x_in.device)
                        if bbox.blend_mode == BlendMode.REMOVALFOREGROUND:
                            #define a new silcer for the edited region, while the None region in the other batch is not updated to the results
                            removal_slicer = list(bbox.slicer)
                            removal_slicer[0] = slice(0,1)
                            removal_slicer = tuple(removal_slicer)
                            if self.none_region_insource:
                                x_feather_buffer     [removal_slicer] += x_tile_out[0:1,...]
                                x_feather_pred_buffer[removal_slicer] += x_tile_pred[0:1,...]
                                x_feather_mask       [removal_slicer] += bbox.feather_mask
                                x_feather_count      [removal_slicer] += 1
                            else:

                                x_feather_buffer     [bbox.slicer] += x_tile_out
                                x_feather_pred_buffer[bbox.slicer] += x_tile_pred
                                x_feather_mask       [bbox.slicer] += bbox.feather_mask
                                x_feather_count      [bbox.slicer] += 1
                                
                            

                        elif bbox.blend_mode == BlendMode.ADDITIONFOREGROUND:
                            #define a new silcer for the edited region, while the None region in the other batch is not updated to the results
                            addition_slicer = list(bbox.slicer)
                            addition_slicer[0] = slice(1,2)
                            addition_slicer = tuple(addition_slicer)

                            if self.none_region_insource:
                                x_feather_buffer     [addition_slicer] += x_tile_out[1:2,...]
                                x_feather_pred_buffer[addition_slicer] += x_tile_pred[1:2,...]
                                x_feather_mask       [addition_slicer] += bbox.feather_mask
                                x_feather_count      [addition_slicer] += 1
                            else:
                                x_feather_buffer     [bbox.slicer] += x_tile_out
                                x_feather_pred_buffer[bbox.slicer] += x_tile_pred
                                x_feather_mask       [bbox.slicer] += bbox.feather_mask
                                x_feather_count      [bbox.slicer] += 1

                        else:
                
                            x_feather_buffer     [bbox.slicer] += x_tile_out
                            x_feather_pred_buffer[bbox.slicer] += x_tile_pred
                            x_feather_mask       [bbox.slicer] += bbox.feather_mask
                            x_feather_count      [bbox.slicer] += 1

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.deactivate(self.p, bbox.extra_network_data)

                # update progress bar
                self.update_pbar()
        

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        if self.is_ddim:
            x_pred_out = torch.where(self.weights > 1, self.x_pred_buffer / self.weights, self.x_pred_buffer)
            
        # Foreground Feather blending
        if x_feather_buffer is not None:
            # Average overlapping feathered regions
            x_feather_buffer = torch.where(x_feather_count > 1, x_feather_buffer / x_feather_count, x_feather_buffer)
            x_feather_mask   = torch.where(x_feather_count > 1, x_feather_mask   / x_feather_count, x_feather_mask)
            # Weighted average with original x_buffer
            x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask, x_out)
            if self.is_ddim:
                x_feather_pred_buffer = torch.where(x_feather_count > 1, x_feather_pred_buffer / x_feather_count, x_feather_pred_buffer)
                x_pred_out            = torch.where(x_feather_count > 0, x_pred_out * (1 - x_feather_mask) + x_feather_pred_buffer * x_feather_mask, x_pred_out)
        

        # save the intermediate x_0 ,x_t-1 and grad map to analysis
        if kwargs['index'] not in self.map_bank.keys(): self.map_bank[kwargs['index']] = {}
        self.map_bank[kwargs['index']]['x_0'] = x_out.detach().cpu().numpy()
        self.map_bank[kwargs['index']]['x_t-1'] = x_pred_out.detach().cpu().numpy()


        #input x_prev, x_0, output: similarity_map_loss
        #define a loss function to make a guidance to the diffusion step
        # if bbox.blend_mode == BlendMode.WHOLE:
        if kwargs['index'] == self.guide_step[3] * self.p.steps: torch.cuda.empty_cache()

        longer_id = 0 if len(bbox.key_region_ids[0]) >= len(bbox.key_region_ids[1]) else 1
        shorter_id = 1 - longer_id

        # aggregate and obtain the mean cross-attn and self-attn, then select the much more solid ones
        if kwargs['index'] % 5 == 4 and self.kssa and kwargs['index'] <= self.guide_step[3] * self.p.steps:
            whole_region_id = len(self.custom_bboxes) -1
            self.controller.select_region(whole_region_id)
            attns, de_words, word_indexs = self.controller.cal_cross_attn(whole_region_id, bbox.batch_prompts)
            
            #select the solider attns
            # selection 1 : by the variance in the batch
            base_prompts = self.p.all_prompts[0].split(',')
            region_prompts = bbox.prompt[longer_id].split(',')
            sub_region_ids = bbox.region_ids
            # calculate per-region words and there length and the index
            total_prompts = base_prompts+region_prompts

            region_ids = [-1 for i in range(len(base_prompts))] + sub_region_ids[longer_id]

            de_ids = []
            cur_id = 0 # the ptr for the cur id in bbox.prompt
            cur_word = total_prompts[cur_id].lower()
            max_length = len(total_prompts)
            cur_de_id = 0
            while cur_de_id < len(de_words[longer_id]):
                cur_de_word = de_words[longer_id][cur_de_id]
                if cur_de_word == '<|startoftext|>' or cur_de_word == '<|endoftext|>':
                    cur_de_id += 1
                    de_ids.append(-2)
                    continue
                if  cur_de_word == cur_word:
                    de_ids.append(region_ids[cur_id])
                    cur_id += 1
                    if cur_id == max_length: break
                    cur_word = total_prompts[cur_id].lower()
                    cur_de_id += 1
                else:
                    if cur_de_word in cur_word:
                        s_index = cur_word.index(cur_de_word)
                        de_ids.append(region_ids[cur_id])
                        cur_de_id += 1
                        cur_word = cur_word[s_index+len(cur_de_word):]
                    else:
                        # print(cur_word)
                        cur_id += 1
                        if cur_id == max_length: break
                        cur_word = total_prompts[cur_id].lower()
            # cal the variance [[per-region]]
            self.addi_region_words = {longer_id:{}, shorter_id:{}}
            metrics = {longer_id:{}, shorter_id:{}}
            short_length = len(de_words[shorter_id])
            self_attn_index = []
                #三个等长的序列 attns de_ids de_words
            for vi in range(len(de_ids)):
                cur_id = de_ids[vi]
                if cur_id < 0: 
                    continue
                metric_method = self.metric

                if vi < short_length:
                    if cur_id in bbox.region_key_words[shorter_id].keys():
                        if cur_id not in metrics[shorter_id].keys(): metrics[shorter_id][cur_id] = {'index':[],'metric':[]}

                        metric = self.get_metric_for_key_word(attns[shorter_id][vi], metric_method)
                        # (mean, stddv) = cv2.meanStdDev(attns[shorter_id][vi])
                        metrics[shorter_id][cur_id]['metric'].append(metric)
                        metrics[shorter_id][cur_id]['index'].append(vi)

                if cur_id not in metrics[longer_id].keys(): metrics[longer_id][cur_id] = {'index':[],'metric':[]}
                
                metric = self.get_metric_for_key_word(attns[longer_id][vi], metric_method)
                metrics[longer_id][cur_id]['metric'].append(metric)
                metrics[longer_id][cur_id]['index'].append(vi)

            # select the word [[per-region]]

            self.visual_attns = {longer_id:{}, shorter_id:{}}
            for batch_id, batch_var in metrics.items():
                for reg_id, reg_metric in batch_var.items():
                    if reg_id not in self.addi_region_words[batch_id].keys(): 
                        self.addi_region_words[batch_id][reg_id] = []
                        self.visual_attns[batch_id][reg_id] = []
                     
                    if len(reg_metric['metric']) < self.key_words_nums:
                        cur_key_words = []
                        for mi in range(len(reg_metric['index'])):
                            cur_key_words.append(de_words[batch_id][   reg_metric['index'][mi]])
                            if batch_id == 0:
                                self_attn_index.append(word_indexs[batch_id][reg_metric['index'][mi]])
                                
                            self.visual_attns[batch_id][reg_id].append(attns[batch_id][   reg_metric['index'][mi]])
                        self.addi_region_words[batch_id][reg_id] = cur_key_words
                        
                    else:
                        cur_var_index = np.argsort(reg_metric['metric'])
                        cur_key_words = []
                        for mi in range(self.key_words_nums):
                            cur_key_words.append(de_words[batch_id][   reg_metric['index'][cur_var_index[mi]]   ])
                            if batch_id == 0:
                                self_attn_index.append(word_indexs[batch_id][reg_metric['index'][cur_var_index[mi]]])

                            self.visual_attns[batch_id][reg_id].append(attns[batch_id][reg_metric['index'][cur_var_index[mi]]])

                        self.addi_region_words[batch_id][reg_id] = cur_key_words
            
            self.controller.key_cross_attn_index = self_attn_index
        self.controller.key_self_attn_index.clear()

        if kwargs['index'] <= self.guide_step[0] * self.p.steps:
            words = [cur_prompt.split(',') for cur_prompt in bbox.key_prompt]
            
            loss = 0

            with torch.enable_grad():
                x_hat = x_pred_out.detach().requires_grad_()
                #semantic alignment module
                word_sim_maps = {}
                with devices.autocast(True): #disable=img.dtype == devices.dtype_vae
                    
                    set_requires_grad(self.sampler.sampler.model.first_stage_model.decoder, True)
                    x_hat = 1. / self.sampler.sampler.model.scale_factor * x_hat.to(dtype=devices.dtype_vae)
                    imgs = [self.sampler.sampler.model.first_stage_model.decode(x_hat[i:i+1])[0].requires_grad_() for i in range(x_hat.size(0))]

                if False:
                    # single-word level alignment###########################################################
                    for bi in range(len(imgs)):
                        word_sim_maps[bi] = []
                        for pi in range(len(words[bi])):
                            preds = self.guidemodel(imgs[bi].repeat(1,1,1,1).float(), words[bi][pi])[0]
                            word_sim_maps[bi].append(torch.sigmoid(preds[0][0]))
                    
                    #region-wise independance loss
                    for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
                        out_of_region_mask = 1 - self.custom_bboxes[region_id].region_mask
                        if ri < len(bbox.key_region_ids[shorter_id]):
                            loss += torch.mean(word_sim_maps[shorter_id][ri] * out_of_region_mask)
                        loss += torch.mean(word_sim_maps[longer_id][ri] * out_of_region_mask)
                else:
                    # regional-word level alignment###########################################################
                    region_words = {longer_id:{}, shorter_id:{}}
                    # for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
                    for region_id, region_word_ in bbox.region_key_words[longer_id].items():
                        if region_id in bbox.region_key_words[shorter_id].keys():
                            if region_id not in region_words[shorter_id].keys(): region_words[shorter_id][region_id] = []
                            
                            region_words[shorter_id][region_id] = bbox.region_key_words[shorter_id][region_id].copy()
                            if self.addi_region_words[shorter_id] != {}:
                                region_words[shorter_id][region_id] += self.addi_region_words[shorter_id][region_id]
                        if region_id not in region_words[longer_id].keys(): region_words[longer_id][region_id] = []
                        region_words[longer_id][region_id] = bbox.region_key_words[longer_id][region_id].copy()
                        if self.addi_region_words[longer_id] != {}:
                            region_words[longer_id][region_id] += self.addi_region_words[longer_id][region_id]
                    
                    region_id_of_maps = [[], []] #保存对每个region进行预测的相似度图的区域号，防止出现非顺序的异常状况
                    for bi in range(len(imgs)):
                        word_sim_maps[bi] = []
                        for region_id, region_word in region_words[bi].items():
                            # if region_id==0: continue
                            preds = self.guidemodel(imgs[bi].repeat(1,1,1,1).float(),  ','.join(region_word))[0]
                            # word_sim_maps[bi][region_id] = torch.sigmoid(preds[0][0])
                            word_sim_maps[bi].append(torch.sigmoid(preds[0][0]))
                            region_id_of_maps[bi].append(region_id)

                    #region-wise independance loss
                    for ri in range(len(word_sim_maps[longer_id])):
                        out_of_region_mask = torch.ones((1, 1, H, W), device=x_in.device)
                        out_of_region_mask[self.custom_bboxes[region_id_of_maps[longer_id][ri]].slicer] -= 1
                        # ！！！！！将-=1这里修改为高斯形式的渐变区域，而不是bbox的硬矩形区域
 
                        # out_of_region_mask = 1 - self.custom_bboxes[ri].region_mask
                        if ri < len(word_sim_maps[shorter_id]):
                            loss += torch.mean(word_sim_maps[shorter_id][ri] * out_of_region_mask[0][0])*self.grad_scale[0]
                        loss += torch.mean(word_sim_maps[longer_id][ri] * out_of_region_mask[0][0])*self.grad_scale[0]

                ##################################################################################

                #object-wise interaction loss
                # for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
                if kwargs['index'] <= self.guide_step[1] * self.p.steps:
                    weight_interregion = self.grad_scale[1] if kwargs['index'] <= 0.5 * self.p.steps else self.grad_scale[1]
                    loss += self.guide_loss(word_sim_maps[shorter_id], word_sim_maps[longer_id]) * weight_interregion

                #background preserve loss
                if kwargs['index'] <= self.guide_step[2] * self.p.steps:

                    ############method 1: in RGB space
                    bg_region_mask = torch.ones((1, 1, H*8, W*8), device=x_in.device)
                    for ri in range(len(word_sim_maps[longer_id])):
                        if region_id_of_maps[longer_id][ri] == 0: continue

                        x = self.custom_bboxes[region_id_of_maps[longer_id][ri]].x*8
                        y = self.custom_bboxes[region_id_of_maps[longer_id][ri]].y*8
                        w = self.custom_bboxes[region_id_of_maps[longer_id][ri]].w*8
                        h = self.custom_bboxes[region_id_of_maps[longer_id][ri]].h*8
                        cur_slicer = slice(None), slice(None), slice(y, y+h), slice(x, x+w)
                        cur_feather_mask = torch.ones((h, w), device=x_in.device)
                        cur_feather_mask = self.feather_mask(w,h,cur_feather_mask)
                        if self.custom_bboxes[region_id_of_maps[longer_id][ri]].blend_mode == BlendMode.FOREGROUND:
                            bg_region_mask[cur_slicer] = cur_feather_mask
                        elif self.custom_bboxes[region_id_of_maps[longer_id][ri]].blend_mode in [BlendMode.ADDITIONFOREGROUND, BlendMode.REMOVALFOREGROUND, BlendMode.EDITEDPAIR]:
                            if not self.soft_preserve: cur_feather_mask[:,:] = 1
                            bg_region_mask[cur_slicer] -= cur_feather_mask
                            # if bg_region_mask[cur_slicer].min() < 0:
                            #     bg_region_mask[cur_slicer] = 0
                            bg_region_mask = torch.clamp(bg_region_mask, min=0)
                    weight_bgp = self.grad_scale[2] if kwargs['index'] > 0.5 * self.p.steps else self.grad_scale[2]/2
                    loss += self.bgpreserve_loss(imgs[0].repeat(1,1,1,1).float() * bg_region_mask, imgs[1].repeat(1,1,1,1).float()*bg_region_mask) * weight_bgp
                # else:
                    ############method 2: in latent noise space

                    bg_region_mask = torch.ones((1, 1, H, W), device=x_in.device)
                    for ri in range(len(word_sim_maps[longer_id])):
                        if region_id_of_maps[longer_id][ri] == 0: continue
                        cur_slicer = self.custom_bboxes[region_id_of_maps[longer_id][ri]].slicer
                        w = self.custom_bboxes[region_id_of_maps[longer_id][ri]].w
                        h = self.custom_bboxes[region_id_of_maps[longer_id][ri]].h
                        cur_feather_mask = torch.ones((h, w), device=x_in.device)
                        cur_feather_mask = self.feather_mask(w,h,cur_feather_mask)
                        if self.custom_bboxes[region_id_of_maps[longer_id][ri]].blend_mode == BlendMode.FOREGROUND:
                            bg_region_mask[cur_slicer] = cur_feather_mask
                        elif self.custom_bboxes[region_id_of_maps[longer_id][ri]].blend_mode in [BlendMode.ADDITIONFOREGROUND, BlendMode.REMOVALFOREGROUND, BlendMode.EDITEDPAIR]:
                            if not self.soft_preserve: cur_feather_mask[:,:] = 1
                            bg_region_mask[cur_slicer] -= cur_feather_mask
                            # if bg_region_mask[cur_slicer].min() < 0:
                            #     bg_region_mask[cur_slicer] = 0
                            bg_region_mask = torch.clamp(bg_region_mask, min=0)
                    weight_bgp = self.grad_scale[2] if kwargs['index'] > 0.5 * self.p.steps else self.grad_scale[2]/2
                    loss += self.bgpreserve_loss(x_hat[0:1,...]*bg_region_mask, x_hat[1:2,...]*bg_region_mask) * weight_bgp
                    if self.flag_noise_inverse:
                        loss += self.bgpreserve_loss(self.all_latents[kwargs['index']]['total']*bg_region_mask, x_hat[0:1,...]*bg_region_mask) * weight_bgp / 10

                grad = -torch.autograd.grad(loss,x_hat)[0]
            #new noise = noise - delta_Loss

            alphas = self.sampler.sampler.ddim_alphas
            alphas_prev = self.sampler.sampler.ddim_alphas_prev
            sqrt_one_minus_alphas = self.sampler.sampler.ddim_sqrt_one_minus_alphas
            sigmas = self.sampler.sampler.ddim_sigmas
            # select parameters corresponding to the currently considered timestep
            b = x_in.shape[0]
            a_t = torch.full((b, 1, 1, 1), alphas[kwargs['index']], device=x_in.device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[kwargs['index']], device=x_in.device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[kwargs['index']], device=x_in.device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[kwargs['index']],device=x_in.device)

            e_t_ = (x_in - a_t.sqrt() * x_pred_out) / sqrt_one_minus_at
            e_t = e_t_ - sqrt_one_minus_at * grad 
            #new_x_0 = xxx
            new_x_0 = (x_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
            #new_x_t-1 = xxx
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

            x_prev = a_prev.sqrt() * new_x_0 + dir_xt
            

            #!!!!!!这里加一个判断，根据不同的置零置空，决定是否更新src img
            if self.opti_source:
                x_out, x_pred_out = x_prev, new_x_0
            else:
                x_out[1:,...], x_pred_out[1:,...] = x_prev[1:,...], new_x_0[1:,...]

            self.map_bank[kwargs['index']]['e_t_'] = e_t_.detach().cpu().numpy()
            self.map_bank[kwargs['index']]['new_x_0'] = x_out.detach().cpu().numpy()
            self.map_bank[kwargs['index']]['new_x_t-1'] = x_pred_out.detach().cpu().numpy()
            self.map_bank[kwargs['index']]['grad'] = grad.detach().cpu().numpy()
            if kwargs['index'] == 0: 
                try:
                    np.save('map_bank-{}-{}-{}-{}-{}-{}.npy'.format(self.grad_scale[0],self.grad_scale[1],self.grad_scale[2], self.guide_step[1], self.guide_step[2], self.guide_step[3]), self.map_bank)
                    torch.save({'map': word_sim_maps, 'words':region_words, 'entro':self.visual_attns, 'select-key':self.addi_region_words}, './outputs/cross-attn/{}-{}-{}-{}.pt'.format(kwargs['index'], self.grad_scale[0],self.grad_scale[1],self.grad_scale[2]))
                except:
                    print('no visual_attns in self')
            #此处可选，x_feather_buffer是继承自原来的，还是用新的，验证实验结果

        #save the final(t_step = 0) cross-attn in the whole region
        # if kwargs['index'] == 0 and self.get_crs_attn:
        #     whole_region_id = len(self.custom_bboxes) -1
        #     self.controller.select_region(whole_region_id)
        #     attns, de_words = self.controller.show_cross_attn(whole_region_id, bbox.batch_prompts)
        #     for ida, attn in enumerate(attns):
        #         cv2.imwrite('./outputs/cross-attn/{ida}.png'.format(ida=str(ida)),attn[:,:10240,:])
        
        
            # for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
            #     if ri < len(bbox.key_region_ids[shorter_id]):
            #         if region_id not in region_words[shorter_id].keys(): region_words[shorter_id][region_id] = []
            #         region_words[shorter_id][region_id].append(words[shorter_id][ri])
            #     if region_id not in region_words[longer_id].keys(): region_words[longer_id][region_id] = []
            #     region_words[longer_id][region_id].append(words[longer_id][ri])
            
            # selection 2: by the inconsistence between the 2 batches
            # for word, region_id in zip(bbox.prompt.split(','), bbox.region_ids):
            #     word

            

        if kwargs['index'] % 5 == 4 and kwargs['index'] <= self.guide_step[0] * self.p.steps:
            try:
                if kwargs['index'] > self.guide_step[3] * self.p.steps:
                    torch.save({'map': word_sim_maps, 'words':region_words}, './outputs/cross-attn/{}-{}-{}-{}.pt'.format(kwargs['index'], self.grad_scale[0],self.grad_scale[1],self.grad_scale[2]))
                else:
                    torch.save({'map': word_sim_maps, 'words':region_words, 'entro':self.visual_attns, 'select-key':self.addi_region_words}, './outputs/cross-attn/{}-{}-{}-{}.pt'.format(kwargs['index'], self.grad_scale[0],self.grad_scale[1],self.grad_scale[2]))
            except:
                print('no visual_attns in self')

        return x_out if self.is_kdiff else (x_out, x_pred_out)
    

    def feather_mask(self, w:int, h:int, mask:Tensor, ratio=0.2) -> Tensor:
        '''Generate a feather mask for the bbox'''
        # mask = torch.ones((h, w), device=x_in.device)
        feather_radius = int(min(w//2, h//2) * ratio)
        # Generate the mask via gaussian weights
        # adjust the weight near the edge. the closer to the edge, the lower the weight
        # weight = ( dist / feather_radius) ** 2

        # 获取矩阵的行数和列数
        rows, cols = mask.shape

        # 创建两个矩阵，分别表示每个元素的行索引和列索引
        row_indices, col_indices = torch.meshgrid(torch.arange(rows), torch.arange(cols))

        # 计算每个元素距离上下边界的距离
        row_distances = torch.min(row_indices, rows - 1 - row_indices)

        # 计算每个元素距离左右边界的距离
        col_distances = torch.min(col_indices, cols - 1 - col_indices)

        # 计算每个元素距离边界的较短的那个距离
        distances = torch.min(row_distances, col_distances)

        # 对距离进行平方运算
        x = (distances / feather_radius) ** 2

        # 定义一个条件，当元素距离矩阵边界距离大于 20 时，返回 0，否则返回元素原来的值
        mask = torch.where(distances.to(device = mask.device) >= feather_radius, mask, x.to(device = mask.device))
        # for i in range(h//2):
        #     for j in range(w//2):
        #         dist = min(i, j)
        #         if dist >= feather_radius: continue
        #         weight = (dist / feather_radius) ** 2
        #         mask[i, j] = weight
        #         mask[i, w-j-1] = weight
        #         mask[h-i-1, j] = weight
        #         mask[h-i-1, w-j-1] = weight

        return mask

    def get_metric_for_key_word(self, img, metric_method):

        if metric_method == 'var':
            (mean, var) = cv2.meanStdDev(img)
            metric = var[0][0]
        elif metric_method == 'entropy1d':
            hist_cv = cv2.calcHist([img[:,:,0]],[0],None,[256],[0,256])
            P = hist_cv/(img.shape[0]*img.shape[1])  #概率
            metric = np.sum([p *np.log2(1/p) if p != 0 else 0 for p in P])[0]
        elif metric_method == 'entropy2d':
            N = 1 # 设置邻域属性，目标点周围1个像素点设置为邻域，九宫格，如果为2就是25宫格...
            S=img.shape
            IJ = []
            #计算j
            for row in range(S[0]):
                for col in range(S[1]):
                    Left_x=np.max([0,col-N])
                    Right_x=np.min([S[1],col+N+1])
                    up_y=np.max([0,row-N])
                    down_y=np.min([S[0],row+N+1])
                    region=img[up_y:down_y,Left_x:Right_x] # 九宫格区域
                    j = (np.sum(region) - img[row][col][0])/((2*N+1)**2-1)
                    IJ.append([img[row][col][0],j])
            # print(IJ)
            # 计算F(i,j)
            F=[]
            arr = [list(i) for i in set(tuple(j) for j in IJ)] #去重，会改变顺序，不过此处不影响
            for i in range(len(arr)):
                F.append(IJ.count(arr[i]))
            # print(F)
            # 计算pij
            P=np.array(F)/(img.shape[0]*img.shape[1])#也是img的W*H
            # 计算熵

            # metric = np.sum([p *np.log2(1/p) for p in P])
            metric = np.sum([p *np.log2(1/p) if p != 0 else 0 for p in P])
        return metric

    def get_noise(self, x_in:Tensor, sigma_in:Tensor, cond_in:Dict[str, Tensor], step:int) -> Tensor:
        # NOTE: The following code is analytically wrong but aesthetically beautiful
        local_cond_in = cond_in.copy()
        def org_func(x:Tensor):
            return shared.sd_model.apply_model(x, sigma_in, cond=local_cond_in)

        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]):
            sigma_in_tile = sigma_in.repeat(len(bboxes))
            new_cond = self.repeat_cond_dict(local_cond_in, bboxes)
            x_tile_out = shared.sd_model.apply_model(x_tile, sigma_in_tile, cond=new_cond)
            return x_tile_out
        
        def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox):
            # The negative prompt in custom bbox should not be used for noise inversion
            # otherwise the result will be astonishingly bad.
            cond = Condition.reconstruct_cond(bbox.cond, step)
            image_cond = self.get_image_cond(local_cond_in)
            if image_cond.shape[2:] == (self.h, self.w):
                image_cond = image_cond[bbox.slicer]
            image_conditioning = image_cond
            self.make_condition_dict([cond], image_conditioning)
            return shared.sd_model.apply_model(x, sigma_in, cond=cond_in)

        return self.sample_one_step(x_in, org_func, repeat_func, custom_func)
    
    @custom_bbox
    def ddim_custom_inversion(self, x:Tensor, cond_in:CondDict, bbox:CustomBBox, bbox_id:int, ts:Tensor, forward_func:Callable, *args, **kwargs) -> Tensor:
        ''' draw custom bbox '''

        tensor, uncond, image_conditioning = self.reconstruct_custom_cond(cond_in, bbox.cond, bbox.uncond, bbox)

        cond = tensor
        # for DDIM, shapes definitely match. So we dont need to do the same thing as in the KDIFF sampler.
        if uncond.shape[1] < cond.shape[1]:
            last_vector = uncond[:, -1:]
            last_vector_repeated = last_vector.repeat([1, cond.shape[1] - uncond.shape[1], 1])
            uncond = torch.hstack([uncond, last_vector_repeated])
        elif uncond.shape[1] > cond.shape[1]:
            uncond = uncond[:, :cond.shape[1]]

        # Wrap the image conditioning back up since the DDIM code can accept the dict directly.
        # Note that they need to be lists because it just concatenates them later.
        if image_conditioning is not None:
            cond   = self.make_condition_dict([cond],  image_conditioning)
            uncond = self.make_condition_dict([uncond], image_conditioning)
        
        # We cannot determine the batch size here for different methods, so delay it to the forward_func.
        return forward_func(x, cond, ts, unconditional_conditioning=uncond, bbox_id = bbox_id, *args, **kwargs)  # lead to the p_sample_ddim in ddim.py


    @keep_signature
    def ddim_inversion(self, x_in:Tensor, cond_in:Union[CondDict, Tensor], ts:Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        '''
        This function will replace the original p_sample_ddim function in ldm/diffusionmodels/ddim.py
        So its signature should be the same as the original function,
        Particularly, the unconditional_conditioning should be with exactly the same name
        '''

        assert VanillaStableDiffusionSampler.p_sample_ddim_hook

        def org_func(x:Tensor):
            return self.sampler_forward(x, cond_in, ts,  *args, **kwargs)

        def repeat_func(x_tile:Tensor, bboxes:List[CustomBBox]):

            new_cond = self.repeat_cond_dict(cond_in, bboxes)
            x_tile_out = shared.sd_model.apply_model(x_tile, new_cond, ts,  *args, **kwargs)
            return x_tile_out

        def custom_func(x:Tensor, bbox_id:int, bbox:CustomBBox):
            # before the final forward, we can set the control tensor
            def forward_func(x, *args, **kwargs):
                self.set_custom_controlnet_tensors(bbox_id, 2*x.shape[0])
                self.set_custom_stablesr_tensors(bbox_id)
                return p_sample_ddim_opitimization(x, *args, **kwargs)
            return self.ddim_custom_inversion(x, cond_in, bbox, bbox_id, ts, forward_func, *args, **kwargs)

        def p_sample_ddim_opitimization(x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None, is_inversion = False, optimization = False,
                    latent_prev = None, num_opti_steps = 10,epsilon = 1e-5, bbox_id = None):
            def set_requires_grad(model, value):
                for param in model.parameters():
                    param.requires_grad = value
            
            b, *_, device = *x.shape, x.device

            alphas = self.sampler.sampler.ddim_alphas
            sqrt_one_minus_alphas = self.sampler.sampler.ddim_sqrt_one_minus_alphas
            sigmas = self.sampler.sampler.ddim_sigmas
            
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)
            
            if optimization:
                assert latent_prev != None and bbox_id != None, "no latent_prev in the null-text optimization"
                if index != 0 or bbox_id == len(self.custom_bboxes)-1:

                    c_in = dict()
                    uc_in = dict()
                    assert isinstance(unconditional_conditioning, dict)
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([c[k][i]]) for i in range(len(c[k]))]
                            uc_in[k] = [torch.cat([unconditional_conditioning[k][i]]) for i in range(len(c[k]))]
                    uc_in['c_crossattn'][0] = uc_in['c_crossattn'][0].clone().detach().requires_grad_()
                    optimizer = torch.optim.Adam(uc_in['c_crossattn'], lr=1e-2 * (1. - (self.p.steps - 1 - index) / 100.))
                    with torch.no_grad():
                        model_t = self.sampler.sampler.model.apply_model(x, t, c_in)
                    with torch.enable_grad():
                        set_requires_grad(self.sampler.sampler.model.model.diffusion_model, True)
                        for j in range(num_opti_steps):
                            model_uncond = self.sampler.sampler.model.apply_model(x, t, uc_in).requires_grad_()
                            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

                            alphas_prev = self.sampler.sampler.ddim_alphas_prev

                            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
                            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
                            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
                            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

                            # current prediction for x_0

                            e_t = model_output

                            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

                            # direction pointing to x_t
                            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

                            latents_prev_rec = a_prev.sqrt() * pred_x0 + dir_xt

                            loss = torch.nn.functional.mse_loss(latents_prev_rec, latent_prev[bbox_id])
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            loss_item = loss.item()
                            if loss_item < epsilon + (self.p.steps - 1 - index) * 2e-5:
                                break
                        # set_requires_grad(self.sampler.sampler.model.model.diffusion_model, False)
                    unconditional_conditioning['c_crossattn'] = uc_in['c_crossattn']
            with torch.no_grad():
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                if isinstance(c, dict):
                    assert isinstance(unconditional_conditioning, dict)
                    c_in = dict()
                    for k in c:
                        if isinstance(c[k], list):
                            c_in[k] = [torch.cat([
                                unconditional_conditioning[k][i],
                                c[k][i]]) for i in range(len(c[k]))]
                        else:
                            c_in[k] = torch.cat([
                                    unconditional_conditioning[k],
                                    c[k]])
                elif isinstance(c, list):
                    c_in = list()
                    assert isinstance(unconditional_conditioning, list)
                    for i in range(len(c)):
                        c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
                else:
                    c_in = torch.cat([unconditional_conditioning, c])
                model_uncond, model_t = self.sampler.sampler.model.apply_model(x_in, t_in, c_in).chunk(2)
                model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)


                if self.sampler.sampler.model.parameterization == "v":
                    e_t = self.sampler.sampler.model.predict_eps_from_z_and_v(x, t, model_output)
                else:
                    e_t = model_output

                if score_corrector is not None:
                    assert self.sampler.sampler.model.parameterization == "eps", 'not implemented'
                    e_t = score_corrector.modify_score(self.sampler.sampler.model, e_t, x, t, c, **corrector_kwargs)

                
                # select parameters corresponding to the currently considered timestep
                if not is_inversion:
                    alphas_prev = self.sampler.sampler.ddim_alphas_prev
                    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
                    # current prediction for x_0
                    if self.sampler.sampler.model.parameterization != "v":
                        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    else:
                        pred_x0 = self.sampler.sampler.model.predict_start_from_z_and_v(x, t, model_output)


                    if dynamic_threshold is not None:
                        raise NotImplementedError()

                    # direction pointing to x_t
                    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

                    x_prev = a_prev.sqrt() * pred_x0 + dir_xt
                    
                    return x_prev, pred_x0, unconditional_conditioning['c_crossattn'][0].detach() if optimization else 0
                else:
                    alphas_next = torch.cat([self.sampler.sampler.ddim_alphas[1:],  self.sampler.sampler.alphas_cumprod[-1:]])
                    a_next= torch.full((b, 1, 1, 1), alphas_next[index], device=device)

                    # current prediction for x_0
                    if self.sampler.sampler.model.parameterization != "v":
                        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    else:
                        pred_x0 = self.sampler.sampler.model.predict_start_from_z_and_v(x, t, model_output)

                    if dynamic_threshold is not None:
                        raise NotImplementedError()

                    # direction pointing to x_t
                    dir_xt = (1. - a_next - sigma_t**2).sqrt() * e_t

                    x_next = a_next.sqrt() * pred_x0 + dir_xt

                    return x_next, pred_x0, 0
            

        return self.inversion_one_step(x_in, org_func, repeat_func, custom_func, *args, **kwargs)

    def inversion_one_step(self, x_in:Tensor, org_func: Callable, repeat_func:Callable, custom_func:Callable, *args, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        '''
        this method splits the whole latent and process in tiles
            - x_in: current whole U-Net latent
            - org_func: original forward function, when use highres
            - denoise_func: one step denoiser for grid tile
            - denoise_custom_func: one step denoiser for custom tile
        '''

        N, C, H, W = x_in.shape
        if H != self.h or W != self.w:
            self.reset_controlnet_tensors()
            return org_func(x_in)

        # clear buffer canvas
        self.reset_buffer(x_in)
        if kwargs['optimization']: uncond_embeddings_list = {}
        if kwargs['is_inversion']: inversion_latents = {}
        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if state.interrupted: return x_in

                # batching
                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])
                x_tile = torch.cat(x_tile_list, dim=0)

                # controlnet tiling
                # FIXME: is_denoise is default to False, however it is set to True in case of MixtureOfDiffusers
                self.switch_controlnet_tensors(batch_id, N, len(bboxes))

                # stablesr tiling
                self.switch_stablesr_tensors(batch_id)


                # compute tiles
                if self.is_kdiff:
                    x_tile_out = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                else:
                    x_tile_out, x_tile_pred = repeat_func(x_tile, bboxes)
                    for i, bbox in enumerate(bboxes):
                        self.x_buffer     [bbox.slicer] += x_tile_out [i*N:(i+1)*N, :, :, :]
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred[i*N:(i+1)*N, :, :, :]

                # update progress bar
                self.update_pbar()

        # Custom region sampling (custom bbox)
        x_feather_buffer      = None
        x_feather_mask        = None
        x_feather_count       = None
        x_feather_pred_buffer = None
        if len(self.custom_bboxes) > 0:
            for bbox_id, bbox in enumerate(self.custom_bboxes):
                #在分区域的region latent noise 具有足够的引导时，才启用WHOLE的交互引导
                if kwargs['index'] > self.guide_step[3] * self.p.steps: 
                    if  bbox.blend_mode == BlendMode.WHOLE: 
                        continue
                # else:
                #     if bbox.blend_mode == BlendMode.ADDITIONFOREGROUND:
                #         continue
                if state.interrupted: return x_in

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.activate(self.p, bbox.extra_network_data)

                x_tile = x_in[bbox.slicer]

                # self.controller.select_region(bbox_id)
                if self.is_kdiff:
                    # retrieve original x_in from construncted input
                    x_tile_out = custom_func(x_tile, bbox_id, bbox)

                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer[bbox.slicer] += x_tile_out
                    else:
                    # elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer = torch.zeros_like(self.x_buffer)
                            x_feather_mask   = torch.zeros((1, 1, H, W), device=x_in.device)
                            x_feather_count  = torch.zeros((1, 1, H, W), device=x_in.device)
                        x_feather_buffer[bbox.slicer] += x_tile_out
                        x_feather_mask  [bbox.slicer] += bbox.feather_mask
                        x_feather_count [bbox.slicer] += 1
                else:
                    #here, in DDIM, apply the original DDIM p_sample, otherwise apply the sample function in K-diff way
                    x_tile_out, x_tile_pred, opti_uncond = custom_func(x_tile, bbox_id, bbox)
                    if kwargs['optimization']:
                        uncond_embeddings_list[bbox_id] = opti_uncond
                    if kwargs['is_inversion']:
                        inversion_latents[bbox_id] = x_tile_out
                    ######################################################33
                    if bbox.blend_mode == BlendMode.BACKGROUND:
                        self.x_buffer     [bbox.slicer] += x_tile_out
                        self.x_pred_buffer[bbox.slicer] += x_tile_pred
                    else:
                    # elif bbox.blend_mode == BlendMode.FOREGROUND:
                        if x_feather_buffer is None:
                            x_feather_buffer      = torch.zeros_like(self.x_buffer)
                            x_feather_pred_buffer = torch.zeros_like(self.x_pred_buffer)
                            x_feather_mask        = torch.zeros((1, 1, H, W), device=x_in.device)
                            x_feather_count       = torch.zeros((1, 1, H, W), device=x_in.device)
                        x_feather_buffer     [bbox.slicer] += x_tile_out
                        x_feather_pred_buffer[bbox.slicer] += x_tile_pred
                        x_feather_mask       [bbox.slicer] += bbox.feather_mask
                        x_feather_count      [bbox.slicer] += 1

                if not self.p.disable_extra_networks:
                    with devices.autocast():
                        extra_networks.deactivate(self.p, bbox.extra_network_data)

                # update progress bar
                self.update_pbar()
        

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)
        if self.is_ddim:
            x_pred_out = torch.where(self.weights > 1, self.x_pred_buffer / self.weights, self.x_pred_buffer)
            
        # Foreground Feather blending
        if x_feather_buffer is not None:
            # Average overlapping feathered regions
            x_feather_buffer = torch.where(x_feather_count > 1, x_feather_buffer / x_feather_count, x_feather_buffer)
            x_feather_mask   = torch.where(x_feather_count > 1, x_feather_mask   / x_feather_count, x_feather_mask)
            # Weighted average with original x_buffer
            x_out = torch.where(x_feather_count > 0, x_out * (1 - x_feather_mask) + x_feather_buffer * x_feather_mask, x_out)
            if self.is_ddim:
                x_feather_pred_buffer = torch.where(x_feather_count > 1, x_feather_pred_buffer / x_feather_count, x_feather_pred_buffer)
                x_pred_out            = torch.where(x_feather_count > 0, x_pred_out * (1 - x_feather_mask) + x_feather_pred_buffer * x_feather_mask, x_pred_out)
        
        

        if kwargs['optimization']:
            return (x_out, uncond_embeddings_list) if self.is_kdiff else (x_out, x_pred_out, uncond_embeddings_list)
        elif kwargs['is_inversion']:
            inversion_latents['total'] = x_out
            inversion_latents['pre'] = x_pred_out

            return x_out if self.is_kdiff else inversion_latents
        else:
            return x_out if self.is_kdiff else (x_out, x_pred_out)
   
