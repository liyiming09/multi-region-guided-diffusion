
                    # #input x_prev, x_0, output: similarity_map_loss
                    # #define a loss function to make a guidance to the diffusion step
                    # if bbox.blend_mode == BlendMode.WHOLE:
                    #     words = [cur_prompt.split(',') for cur_prompt in bbox.key_prompt]
                    #     longer_id = 0 if len(bbox.key_region_ids[0]) >= len(bbox.key_region_ids[1]) else 1
                    #     shorter_id = 1 - longer_id
                    #     loss = 0

                    #     with torch.enable_grad():
                    #         x_hat = x_tile_pred.detach().requires_grad_()
                    #         #semantic alignment module
                    #         word_sim_maps = {}
                    #         with devices.autocast(True): #disable=img.dtype == devices.dtype_vae
                                
                    #             set_requires_grad(self.sampler.sampler.model.first_stage_model.decoder, True)
                    #             x_hat = 1. / self.sampler.sampler.model.scale_factor * x_hat.to(dtype=devices.dtype_vae)
                    #             imgs = [self.sampler.sampler.model.first_stage_model.decode(x_hat[i:i+1])[0].requires_grad_() for i in range(x_tile_pred.size(0))]
                    #             # imgs = self.sampler.sampler.model.decode_first_stage(x_hat).requires_grad_() 
                    #             # imgs = [self.sampler.sampler.model.decode_first_stage(x_hat[i:i+1].to(dtype=devices.dtype_vae))[0].requires_grad_() for i in range(x_tile_pred.size(0))]

                            

                    #         if False:
                    #             # single-word level alignment###########################################################
                    #             for bi in range(len(imgs)):
                    #                 word_sim_maps[bi] = []
                    #                 for pi in range(len(words[bi])):
                    #                     preds = self.guidemodel(imgs[bi].repeat(1,1,1,1).float(), words[bi][pi])[0]
                    #                     word_sim_maps[bi].append(torch.sigmoid(preds[0][0]))
                                
                    #             #region-wise independance loss
                    #             for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
                    #                 out_of_region_mask = 1 - self.custom_bboxes[region_id].region_mask
                    #                 if ri < len(bbox.key_region_ids[shorter_id]):
                    #                     loss += torch.mean(word_sim_maps[shorter_id][ri] * out_of_region_mask)
                    #                 loss += torch.mean(word_sim_maps[longer_id][ri] * out_of_region_mask)
                    #         else:
                    #             # regional-word level alignment###########################################################
                    #             region_words = {longer_id:{}, shorter_id:{}}
                    #             for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
                    #                 if ri < len(bbox.key_region_ids[shorter_id]):
                    #                     if region_id not in region_words[shorter_id].keys(): region_words[shorter_id][region_id] = []
                    #                     region_words[shorter_id][region_id].append(words[shorter_id][ri])
                    #                 if region_id not in region_words[longer_id].keys(): region_words[longer_id][region_id] = []
                    #                 region_words[longer_id][region_id].append(words[longer_id][ri])
                                
                    #             region_id_of_maps = [[], []] #保存对每个region进行预测的相似度图的区域号，防止出现非顺序的异常状况
                    #             for bi in range(len(imgs)):
                    #                 word_sim_maps[bi] = []
                    #                 for region_id, region_word in region_words[bi].items():
                    #                     if region_id==0: continue
                    #                     preds = self.guidemodel(imgs[bi].repeat(1,1,1,1).float(),  ','.join(region_word))[0]
                    #                     # word_sim_maps[bi][region_id] = torch.sigmoid(preds[0][0])
                    #                     word_sim_maps[bi].append(torch.sigmoid(preds[0][0]))
                    #                     region_id_of_maps[bi].append(region_id)

                    #             #region-wise independance loss
                    #             for ri in range(len(word_sim_maps[longer_id])):
                    #                 out_of_region_mask = torch.ones((1, 1, H, W), device=x_in.device)
                    #                 out_of_region_mask[self.custom_bboxes[region_id_of_maps[longer_id][ri]].slicer] -= 1
                    #                 # out_of_region_mask = 1 - self.custom_bboxes[ri].region_mask
                    #                 if ri < len(word_sim_maps[shorter_id]):
                    #                     loss += torch.mean(word_sim_maps[shorter_id][ri] * out_of_region_mask[0][0])*self.grad_scale[0]
                    #                 loss += torch.mean(word_sim_maps[longer_id][ri] * out_of_region_mask[0][0])*self.grad_scale[0]

                    #         ##################################################################################

                    #         #object-wise interaction loss
                    #         # for ri, region_id in enumerate(bbox.key_region_ids[longer_id]):
                    #         loss += self.guideloss(word_sim_maps[shorter_id], word_sim_maps[longer_id]) *self.grad_scale[1]
                    #         grad = -torch.autograd.grad(loss,x_hat)[0]
                    #     #new noise = noise - delta_Loss

                    #     alphas = self.sampler.sampler.ddim_alphas
                    #     alphas_prev = self.sampler.sampler.ddim_alphas_prev
                    #     sqrt_one_minus_alphas = self.sampler.sampler.ddim_sqrt_one_minus_alphas
                    #     sigmas = self.sampler.sampler.ddim_sigmas
                    #     # select parameters corresponding to the currently considered timestep
                    #     b = x_in.shape[0]
                    #     a_t = torch.full((b, 1, 1, 1), alphas[kwargs['index']], device=x_in.device)
                    #     a_prev = torch.full((b, 1, 1, 1), alphas_prev[kwargs['index']], device=x_in.device)
                    #     sigma_t = torch.full((b, 1, 1, 1), sigmas[kwargs['index']], device=x_in.device)
                    #     sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[kwargs['index']],device=x_in.device)

                    #     e_t_ = (x_in - a_t.sqrt() * x_tile_pred) / sqrt_one_minus_at
                    #     e_t = e_t_ - sqrt_one_minus_at * grad 
                    #     #new_x_0 = xxx
                    #     new_x_0 = (x_in - sqrt_one_minus_at * e_t) / a_t.sqrt()
                    #     #new_x_t-1 = xxx
                    #     # direction pointing to x_t
                    #     dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t

                    #     x_prev = a_prev.sqrt() * new_x_0 + dir_xt
                        
                    #     x_tile_out, x_tile_pred = x_prev, new_x_0

