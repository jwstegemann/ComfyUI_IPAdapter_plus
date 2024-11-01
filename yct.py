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
from copy import deepcopy

try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T

from .image_proj_models import MLPProjModel, MLPProjModelFaceId, ProjModelFaceIdPlus, Resampler, ImageProjModel

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
            is_kwai_kolors=False,
            encoder_hid_proj=None,
            weight_kolors=False
        ).to(device, dtype=dtype)

        del ipadapter

        return (ipa, )


class CreateIPAdapter():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
            },
            "optional": {
            }
        }

    CATEGORY = "ipadapter/plus"
    RETURN_TYPES = ("IPADAPTERINSTANCE", )
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, ipadapter):
        from .IPAdapterPlus import IPAdapter

        if ipadapter is None:
            raise Exception("Missing IPAdapter model.")
        
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        cross_attention_dim = 1280 # if (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm else output_cross_attention_dim
        clip_extra_context_tokens = 16 # if (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm else 4

        # img_cond_embeds = embeds['img_cond_embeds'].to(device, dtype=dtype)

        ipa = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=1280, # img_cond_embeds.shape[-1],
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=True,
            is_plus=True,
            is_full=False,
            is_faceid=False,
            is_portrait_unnorm=False,
            is_kwai_kolors=False,
            encoder_hid_proj=None,
            weight_kolors=False
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
        from .IPAdapterPlus import set_model_patch_replace, weights_unstyled

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

        cond_alt = embeds['cond_alt'] if embeds['cond_alt'] is not None else None # None
        if cond_alt:
            cond_alt[3].to(device, dtype=dtype)

        work_model = model.clone()

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)
        if (weights):
            weight={1:weight * weights[1], 2: weight * weights[2], 3: weight * weights[3], 4: weight * weights[4], 5:weight * weights[5], 6: weight * weights[6], 7: weight * weights[7], 8: weight * weights[8], 9: weight * weights[9], 10: weight * weights[10], 11: weight * weights[11]}
        elif (weight_type == "unstyled"):
            weight={1:weight * weights_unstyled[1], 2: weight * weights_unstyled[2], 3: weight * weights_unstyled[3], 4: weight * weights_unstyled[4], 5:weight * weights_unstyled[5], 6: weight * weights_unstyled[6], 7: weight * weights_unstyled[7], 8: weight * weights_unstyled[8], 9: weight * weights_unstyled[9], 10: weight * weights_unstyled[10], 11: weight * weights_unstyled[11]}

        patch_kwargs = {
            "ipadapter": ipadapterinstance,
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
        
        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 1, index))
            number += 1

        return (work_model, )



class ApplyCompositionAndStyleIPAdapter():
    @classmethod
    def INPUT_TYPES(s):
        from .IPAdapterPlus import WEIGHT_TYPES

        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapterinstance": ("IPADAPTERINSTANCE", ),
                "embeds": ("EMBEDS", ),
                "weight_style": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "weight_composition": ("FLOAT", { "default": 1.0, "min": 0, "max": 5, "step": 0.05 }),
                "expand_style": ("BOOLEAN", { "default": False }),                
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    CATEGORY = "ipadapter/plus"
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"

    def apply_ipadapter(self, model, ipadapterinstance, embeds, weight_style, weight_composition, expand_style, start_at=0.0, end_at=1.0, embeds_scaling='V only', attn_mask=None):
        from .IPAdapterPlus import set_model_patch_replace, weights_unstyled

        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            dtype = torch.float16 if comfy.model_management.should_use_fp16() else torch.float32

        ipadapterinstance.to(device, dtype=dtype)

        weight = weight_style
        weight_type = "strong style and composition" if expand_style else "style and composition"

        if weight_type == "style and composition":
            weight = { 3:weight_composition, 6:weight }
        elif weight_type == "strong style and composition":
            weight = { 0:weight, 1:weight, 2:weight, 3:weight_composition, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }

        if attn_mask is not None:
            attn_mask = attn_mask.to(device, dtype=dtype)

        cond = embeds['cond'].to(device, dtype=dtype) if embeds['cond'] is not None else None # ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2)
        # TODO: check if noise helps with the uncond face embeds
        uncond = embeds['uncond'].to(device, dtype=dtype) if embeds['uncond'] is not None else None # ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2)
        cond_alt = embeds['cond_alt'] if embeds['cond_alt'] is not None else None # None
        if cond_alt:
            cond_alt[3].to(device, dtype=dtype)
        # if img_comp_cond_embeds is not None:
        #     cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

        work_model = model.clone()

        sigma_start = work_model.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = work_model.get_model_object("model_sampling").percent_to_sigma(end_at)

        patch_kwargs = {
            "ipadapter": ipadapterinstance,
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

        number = 0
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
            number += 1

        return (work_model, )
