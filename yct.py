import torch
import os
import math
import folder_paths

import comfy.model_management as model_management
from node_helpers import conditioning_set_values
from comfy.clip_vision import load as load_clip_vision
from comfy.sd import load_lora_for_models
import comfy.utils

import torch.nn as nn
from PIL import Image

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from .image_proj_models import MLPProjModel, MLPProjModelFaceId, ProjModelFaceIdPlus, Resampler, ImageProjModel
from .CrossAttentionPatch import CrossAttentionPatch
from .utils import (
    encode_image_masked,
    tensor_to_size,
    contrast_adaptive_sharpening,
    tensor_to_image,
    image_to_tensor,
    ipadapter_model_loader,
    insightface_loader,
    get_clipvision_file,
    get_ipadapter_file,
    get_lora_file,
)

import time


# comics

class FacePlusIPAdapterFromEmbeds():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "embeds": ("EMBEDS", ),
            },
            "optional": {
            }
        }

    CATEGORY = "ipadapter/plus"
    RETURN_TYPES = ("IPADAPTERINSTANCE", )
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, ipadapter, embeds):
        from .IPAdapterPlus import IPAdapter

        # print("in embeds: ", embeds)

        if ipadapter is None:
            raise Exception("Missing IPAdapter model.")
        
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        cross_attention_dim = 1280 # if (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm else output_cross_attention_dim
        clip_extra_context_tokens = 16 # if (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm else 4

        img_cond_embeds = embeds['img_cond_embeds'].to(device, dtype=dtype)

        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=img_cond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=True,
            is_plus=True,
            is_full=False,
            is_faceid=False,
            is_portrait_unnorm=False,
        ).to(device, dtype=dtype)

        del ipadapter

        return (ipa, )



class FacePlusWeights():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "weight1": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight2": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight3": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight4": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight5": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight6": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight7": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight8": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight9": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight10": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight11": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
            }
        }

    CATEGORY = "ipadapter/plus"
    RETURN_TYPES = ("IPADAPTERWEIGHTS",)
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, weight1,weight2,weight3,weight4,weight5,weight6,weight7,weight8,weight9,weight10,weight11):
        weight={1:weight1, 2: weight2, 3: weight3, 4: weight4, 5:weight5, 6: weight6, 7: weight7, 8: weight8, 9: weight9, 10: weight10, 11: weight11}
        return (weight, )


weight_unstyled = { 1: 0.7, 2: 0.7, 3: 0.98, 4: 0.5, 5: 0.5, 6: 0.25, 7: 0.7, 8: 0.8, 9: 0.85, 10: 0.9, 11: 0.95 }
weight_unstyled_high = { 1: 0.05, 2: 0.05, 3: 0.95, 4: 0.9, 5: 0.9, 6: 0.3, 7: 0.7, 8: 0.8, 9: 0.85, 10: 0.9, 11: 0.95 }

class ApplyFacePlusIPAdapter():
    @classmethod
    def INPUT_TYPES(s):
        from .IPAdapterPlus import WEIGHT_TYPES

        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapterinstance": ("IPADAPTERINSTANCE", ),
                "embeds": ("EMBEDS", ),
                "weight": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "weights": ("IPADAPTERWEIGHTS",)
            }
        }

    CATEGORY = "ipadapter/plus"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, model, ipadapterinstance, embeds, weight, weight_type="linear", start_at=0.0, end_at=1.0, embeds_scaling='V only', attn_mask=None, weights=None):
        from .IPAdapterPlus import set_model_patch_replace

        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        ipadapterinstance.to(device, dtype=dtype)

        if isinstance(weight, list):
            weight = weight[0]

        if attn_mask is not None:
            attn_mask = attn_mask.to(device, dtype=dtype)

        cond = embeds['cond'].to(device, dtype=dtype) if embeds['cond'] is not None else None # ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2)
        # TODO: check if noise helps with the uncond face embeds
        uncond = embeds['uncond'].to(device, dtype=dtype) if embeds['uncond'] is not None else None # ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2)

        cond_alt = embeds['cond_alt'].to(device, dtype=dtype) if embeds['cond_alt'] is not None else None # None
        # if img_comp_cond_embeds is not None:
        #     cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

        work_model = model.clone()

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)
        if (weights):
            weight={1:weight * weights[1], 2: weight * weights[2], 3: weight * weights[3], 4: weight * weights[4], 5:weight * weights[5], 6: weight * weights[6], 7: weight * weights[7], 8: weight * weights[8], 9: weight * weights[9], 10: weight * weights[10], 11: weight * weights[11]}
        elif (weight_type == "unstyled high likeliness"):
            weight={1:weight * weight_unstyled_high[1], 2: weight * weight_unstyled_high[2], 3: weight * weight_unstyled_high[3], 4: weight * weight_unstyled_high[4], 5:weight * weight_unstyled_high[5], 6: weight * weight_unstyled_high[6], 7: weight * weight_unstyled_high[7], 8: weight * weight_unstyled_high[8], 9: weight * weight_unstyled_high[9], 10: weight * weight_unstyled_high[10], 11: weight * weight_unstyled_high[11]}
        elif (weight_type == "unstyled"):
            weight={1:weight * weight_unstyled[1], 2: weight * weight_unstyled[2], 3: weight * weight_unstyled[3], 4: weight * weight_unstyled[4], 5:weight * weight_unstyled[5], 6: weight * weight_unstyled[6], 7: weight * weight_unstyled[7], 8: weight * weight_unstyled[8], 9: weight * weight_unstyled[9], 10: weight * weight_unstyled[10], 11: weight * weight_unstyled[11]}

        patch_kwargs = {
            "ipadapter": ipadapterinstance,
            "number": 0,
            "weight": weight,
            "cond": cond,
            "cond_alt": cond_alt,
            "uncond": uncond,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": False,
            "embeds_scaling": embeds_scaling,
        }

        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                patch_kwargs["number"] += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                patch_kwargs["number"] += 1
        for index in range(10):
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
            patch_kwargs["number"] += 1

        return (work_model, )


#
# FaceID
#


class FaceIDv2IPAdapterXL():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "faceid": ("FACEID", ),
            },
            "optional": {
            }
        }

    CATEGORY = "ipadapter/faceid"
    RETURN_TYPES = ("IPADAPTERINSTANCE", )
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, ipadapter, faceid):
        from .IPAdapterPlus import IPAdapter

        is_sdxl = True

        start_time = time.time()
        #ipadapter_model = ipadapter['ipadapter']['model']

        # from execute
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32
        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]

        is_full = False
        is_portrait = False
        is_portrait_unnorm = False
        is_faceid =  True
        is_plus = True
        is_faceidv2 = True

        cross_attention_dim = 1280 if (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm else output_cross_attention_dim
        clip_extra_context_tokens = 16 if (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm else 4

#        print("before faceid #### ", ((time.time() - start_time) * 1000), "ms.")

        img_cond_embeds = faceid['img_cond_embeds'].to(device, dtype=dtype) if faceid['img_cond_embeds'] is not None else None

#        print("before ipadapter #### ", ((time.time() - start_time) * 1000), "ms.")

        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=img_cond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=is_full,
            is_faceid=is_faceid,
            is_portrait_unnorm=is_portrait_unnorm,
        ).to(device, dtype=dtype)

#        print("after ipadapter #### ", ((time.time() - start_time) * 1000), "ms.")

        return (ipa, )


class ApplyFaceIDv2XL():
    @classmethod
    def INPUT_TYPES(s):
        from .IPAdapterPlus import WEIGHT_TYPES

        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapterinstance": ("IPADAPTERINSTANCE", ),
                "faceid": ("FACEID", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    CATEGORY = "ipadapter/faceid"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, model, ipadapterinstance, faceid, weight=1.0, weight_faceidv2=None, weight_type="linear", start_at=0.0, end_at=1.0, embeds_scaling='V only', attn_mask=None):
        from .IPAdapterPlus import set_model_patch_replace

        is_sdxl = True
#        start_time = time.time()

        # from execute
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        ipadapterinstance.to(device, dtype=dtype)
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

#        print("before ipadapter #### ", ((time.time() - start_time) * 1000), "ms.")

        cond = faceid['cond'].to(device, dtype=dtype) if faceid['cond'] is not None else None # ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2)
        # TODO: check if noise helps with the uncond face embeds
        uncond = faceid['uncond'].to(device, dtype=dtype) if faceid['uncond'] is not None else None # ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2)

        cond_alt = faceid['cond_alt'].to(device, dtype=dtype) if faceid['cond_alt'] is not None else None # None
        # if img_comp_cond_embeds is not None:
        #     cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

        if (weight_type == "unstyled high likeliness"):
            weight={1:weight * weight_unstyled_high[1], 2: weight * weight_unstyled_high[2], 3: weight * weight_unstyled_high[3], 4: weight * weight_unstyled_high[4], 5:weight * weight_unstyled_high[5], 6: weight * weight_unstyled_high[6], 7: weight * weight_unstyled_high[7], 8: weight * weight_unstyled_high[8], 9: weight * weight_unstyled_high[9], 10: weight * weight_unstyled_high[10], 11: weight * weight_unstyled_high[11]}
        elif (weight_type == "unstyled"):
            weight={1:weight * weight_unstyled[1], 2: weight * weight_unstyled[2], 3: weight * weight_unstyled[3], 4: weight * weight_unstyled[4], 5:weight * weight_unstyled[5], 6: weight * weight_unstyled[6], 7: weight * weight_unstyled[7], 8: weight * weight_unstyled[8], 9: weight * weight_unstyled[9], 10: weight * weight_unstyled[10], 11: weight * weight_unstyled[11]}

        work_model = model.clone()

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

#        print("before attn mask #### ", ((time.time() - start_time) * 1000), "ms.")

        if attn_mask is not None:
            attn_mask = attn_mask.to(device, dtype=dtype)

        patch_kwargs = {
            "ipadapter": ipadapterinstance,
            "number": 0,
            "weight": weight,
            "cond": cond,
            "cond_alt": cond_alt,
            "uncond": uncond,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": False,
            "embeds_scaling": embeds_scaling,
        }


        if not is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                patch_kwargs["number"] += 1

#        print(" #### ", ((time.time() - start_time) * 1000), "ms.")
        return (work_model, )


class IPAdapterFromFaceID():
    @classmethod
    def INPUT_TYPES(s):
        from .IPAdapterPlus import WEIGHT_TYPES
        
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "faceid": ("FACEID", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    CATEGORY = "ipadapter/faceid"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, model, ipadapter, faceid, weight=1.0, weight_faceidv2=None, weight_type="linear", combine_embeds="concat", start_at=0.0, end_at=1.0, embeds_scaling='V only', attn_mask=None, clip_vision=None, insightface=None):
        from .IPAdapterPlus import (IPAdapter, set_model_patch_replace)

        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

#        start_time = time.time()

        if 'ipadapter' in ipadapter:
#            print("### in A")
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
 #           print("### in B")
            ipadapter_model = ipadapter

        if clip_vision is None:
#            print("### in C")
            raise Exception("Missing CLIPVision model.")

        weight = weight

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter

        # from execute
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        is_full = "proj.3.weight" in ipadapter_model["image_proj"]
#        print("is_full:", is_full)
        is_portrait = "proj.2.weight" in ipadapter_model["image_proj"] and not "proj.3.weight" in ipadapter_model["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter_model["ip_adapter"]
#        print("is_porrait:", is_full)
        is_portrait_unnorm = "portraitunnorm" in ipadapter_model
#        print("portrait_unnorm:", is_portrait_unnorm)
        is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter_model["ip_adapter"] or is_portrait_unnorm
#        print("is_faceid:", is_faceid)
        is_plus = (is_full or "latents" in ipadapter_model["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]) and not is_portrait_unnorm
#        print("is_plus:", is_plus)
        is_faceidv2 = "faceidplusv2" in ipadapter_model
#        print("is_faceidv2:", is_faceidv2)
        output_cross_attention_dim = ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]
#        print("output_cross_attention_dim", output_cross_attention_dim)

        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

        cross_attention_dim = 1280 if (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm else output_cross_attention_dim
        clip_extra_context_tokens = 16 if (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm else 4



#        print("before faceid #### ", ((time.time() - start_time) * 1000), "ms.")

        img_cond_embeds = faceid['img_cond_embeds'].to(device, dtype=dtype) if faceid['img_cond_embeds'] is not None else None

#        print("before ipadapter #### ", ((time.time() - start_time) * 1000), "ms.")

        ipa = IPAdapter(
            ipadapter_model,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=img_cond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=is_sdxl,
            is_plus=is_plus,
            is_full=is_full,
            is_faceid=True,
            is_portrait_unnorm=is_portrait_unnorm,
        ).to(device, dtype=dtype)

#        print("after ipadapter #### ", ((time.time() - start_time) * 1000), "ms.")


        cond = faceid['cond'].to(device, dtype=dtype) if faceid['cond'] is not None else None # ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2)
        # TODO: check if noise helps with the uncond face embeds
        uncond = faceid['uncond'].to(device, dtype=dtype) if faceid['uncond'] is not None else None # ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2)

        cond_alt = faceid['cond_alt'].to(device, dtype=dtype) if faceid['cond_alt'] is not None else None # None
        # if img_comp_cond_embeds is not None:
        #     cond_alt = { 3: cond_comp.to(device, dtype=dtype) }


#        print("before mordel  #### ", ((time.time() - start_time) * 1000), "ms.")

        work_model = model.clone()

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)


#        print("before attn mask #### ", ((time.time() - start_time) * 1000), "ms.")

        if attn_mask is not None:
            attn_mask = attn_mask.to(device, dtype=dtype)

        patch_kwargs = {
            "ipadapter": ipa,
            "number": 0,
            "weight": weight,
            "cond": cond,
            "cond_alt": cond_alt,
            "uncond": uncond,
            "weight_type": weight_type,
            "mask": attn_mask,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "unfold_batch": False,
            "embeds_scaling": embeds_scaling,
        }


        if not is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                patch_kwargs["number"] += 1

        del ipadapter_model

#        print(" #### ", ((time.time() - start_time) * 1000), "ms.")
        return (work_model, None)


#
# here Storing, Loading and Reusing FaceID
#
class IPAdapterSaveFaceId:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "faceid": ("FACEID",),
            "filename": ("STRING", {"default": "FaceID"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter/faceid"

    def save(self, faceid, filename):
        local_path = os.path.join(folder_paths.get_output_directory(), filename)
        torch.save(faceid, local_path)
        return (None, )


class IPAdapterLoadFaceId:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"faceid": ("STRING", {"default": "PathToFaceID"}) } }

    RETURN_TYPES = ("FACEID", )
    FUNCTION = "load"
    CATEGORY = "ipadapter/faceid"

    def load(self, faceid):
        input_dir = folder_paths.get_input_directory()
        path = os.path.join(input_dir, faceid)
        faceid = torch.load(path)
        return ({ "cond": faceid["cond"] , "uncond": faceid["uncond"], "cond_alt" : faceid["cond_alt"], "img_cond_embeds": faceid["img_cond_embeds"]}, )
