# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import time

from tqdm import tqdm
import numpy as np
import torch
from .Sapiens_Pytorch import SapiensPredictor, SapiensConfig
from .Sapiens_Pytorch.classes_and_palettes import GOLIATH_CLASSES_FIX, GOLIATH_CLASSES
from .utils import get_models_path, tensor2cv, load_images, tensor2pil
import folder_paths

# add checkpoints dir
weigths_current_path = os.path.join(folder_paths.models_dir, "sapiens")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)

try:
    folder_paths.add_model_folder_path("sapiens", weigths_current_path, False)
except:
    folder_paths.add_model_folder_path("sapiens", weigths_current_path)

weigths_seg_path = os.path.join(weigths_current_path, "seg")
if not os.path.exists(weigths_seg_path):
    os.makedirs(weigths_seg_path)
weigths_depth_path = os.path.join(weigths_current_path, "depth")
if not os.path.exists(weigths_depth_path):
    os.makedirs(weigths_depth_path)
weigths_pose_path = os.path.join(weigths_current_path, "pose")
if not os.path.exists(weigths_pose_path):
    os.makedirs(weigths_pose_path)
weigths_normal_path = os.path.join(weigths_current_path, "normal")
if not os.path.exists(weigths_normal_path):
    os.makedirs(weigths_normal_path)


class SapiensLoader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_filter = [i for i in folder_paths.get_filename_list("sapiens") if
                            i.endswith(".pth") or i.endswith(".pt2")]
        ckpt_list_seg = [i for i in ckpt_list_filter if "seg" in i]
        ckpt_list_depth = [i for i in ckpt_list_filter if "depth" in i]
        ckpt_list_normal = [i for i in ckpt_list_filter if "normal" in i]
        ckpt_list_pose = [i for i in ckpt_list_filter if "pose" in i]
        return {
            "required": {
                "seg_ckpt": (["none"] + ckpt_list_seg,),
                "depth_ckpt": (["none"] + ckpt_list_depth,),
                "normal_ckpt": (["none"] + ckpt_list_normal,),
                "pose_ckpt": (["none"] + ckpt_list_pose,),
                "dtype": (["float32_torch", "bf16_torch", "float32", "bfloat16", ],),
                "minimum_person_height": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "remove_background": ("BOOLEAN", {"default": True},),
                "use_yolo": ("BOOLEAN", {"default": False},),
                "show_pose_object": ("BOOLEAN", {"default": False},),
                "seg_pellete": ("BOOLEAN", {"default": True},),
                "convert_torchscript_to_bf16": ("BOOLEAN", {"default": False},),
                # currently only for TorchScript models
            },
        }
    
    RETURN_TYPES = ("MODEL_SAPIEN",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "Sapiens"
    
    def loader_main(self, seg_ckpt, depth_ckpt, normal_ckpt, pose_ckpt, dtype, minimum_person_height, remove_background,
                    use_yolo, show_pose_object, seg_pellete, convert_torchscript_to_bf16):
        
        config = SapiensConfig()
        config.model_dir = weigths_current_path
        config.pt_type = dtype
        config.detector = True if use_yolo else None
        config.remove_bg = remove_background
        config.show_pose_object = show_pose_object
        config.use_pellete = seg_pellete
        
        if dtype == "bfloat16":
            config.dtype = torch.bfloat16
        elif dtype == "float32_torch":
            config.dtype = torch.float32
        elif dtype == "bf16_torch":
            config.dtype = torch.bfloat16
        else:
            config.dtype = torch.float32
        
        config.minimum_person_height = minimum_person_height
        config = get_models_path(seg_ckpt, depth_ckpt, normal_ckpt, pose_ckpt, config, dtype)
        
        # currently only for TorchScript models, convert selected FP32 TorchScript Sapiens models to BF16, save them and use them
        if convert_torchscript_to_bf16:
            print("converting TorchScript Sapiens models to BF16...")
            for model_path_attr in ("local_seg_path", "local_depth_path", "local_normal_path", "local_pose_path"):
                model_path = getattr(config, model_path_attr)
                if len(model_path) and model_path.endswith("torchscript.pt2"):
                    model_split_path = os.path.splitext(model_path)
                    converted_model_path = model_split_path[0] + "_bf16" + model_split_path[1]
                    
                    print(f'converting "{model_path}" to BF16...')
                    if os.path.exists(converted_model_path):
                        print(f'"{converted_model_path}" already exists, not converting...')
                    else:
                        model = torch.jit.load(model_path)
                        model.eval().to("cuda").to(torch.bfloat16)
                        torch.jit.save(model, converted_model_path)
                        
                        print(f'"{model_path}" converted to BF16 "{converted_model_path}"')
                    
                    setattr(config, model_path_attr, converted_model_path)
            
            config.dtype = torch.bfloat16
        
        model = SapiensPredictor(config)
        return (model,)


class SapiensSampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_SAPIEN",),
                "image": ("IMAGE",),
                "seg_select": (["none"] + list(GOLIATH_CLASSES_FIX),),
                "add_seg_index": ("STRING", {"default": "", }),
                "save_pose": ("BOOLEAN", {"default": False},),
                "BG_R": ("INT", {
                    "default": 255,
                    "min": 0,  # Minimum value
                    "max": 255,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "slider",  # Cosmetic only: display as "number" or "slider"
                }),
                "BG_G": ("INT", {
                    "default": 255,
                    "min": 0,  # Minimum value
                    "max": 255,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "slider",  # Cosmetic only: display as "number" or "slider"
                }),
                "BG_B": ("INT", {
                    "default": 255,
                    "min": 0,  # Minimum value
                    "max": 255,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "slider",  # Cosmetic only: display as "number" or "slider"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("seg_img", "depth_img", "normal_img", "pose_img", "mask")
    FUNCTION = "sampler_main"
    CATEGORY = "Sapiens"
    
    def sampler_main(self, model, image, seg_select, add_seg_index,save_pose, BG_R, BG_G, BG_B):
        start = time.perf_counter()
        if not torch.backends.mps.is_available():
            if torch.cuda.is_available():
                model.move_to_cuda()
        
        RGB_BG = [BG_R, BG_G, BG_B]
        
        # select body index
        add_seg_list = add_seg_index.split(",") if add_seg_index else []
        add_seg_list = [int(i) for i in add_seg_list if int(i) < 29] if add_seg_list else []
        if seg_select != "none":
            seg_select = seg_select.split(".")[-1]
            seg_select = [list(GOLIATH_CLASSES).index(seg_select)]
            if add_seg_list:
                seg_select = seg_select + add_seg_list  # [0,...,27]
        else:
            seg_select = []
        
        seg_select_all = [list(GOLIATH_CLASSES_FIX)[i] for i in seg_select] if seg_select else "seg default hunman map."
        
        print(f"Select seg part of {seg_select_all} ")
        
        #model.select = seg_select
        
        b, _, _, _ = image.size()
        if b == 1:
            zero_tensor = torch.zeros_like(image, dtype=torch.float32, device="cpu")
            if model.use_pellete:  # pil
                img_in = [tensor2pil(image)]
            else:
                img_in = [tensor2cv(image.squeeze())]
        else:
            image_list = torch.chunk(image, chunks=b)
            zero_tensor = torch.zeros_like(image_list[0], dtype=torch.float32, device="cpu")
            if model.use_pellete:
                img_in = [tensor2pil(i) for i in image_list]  # pil
            else:
                img_in = [tensor2cv(i.squeeze()) for i in image_list]
        seg_list = []
        mask_list = []
        depth_list = []
        normal_list = []
        pose_list = []
        for img in tqdm(img_in):
        #for img in img_in:
            seg, depth, normal, pose, mask = model(img, seg_select, RGB_BG)
            if isinstance(seg, list):
                seg_list.append(seg[0])
                mask_list.append(mask[0])
            if isinstance(depth, list):
                depth_list.append(depth[0])
            if isinstance(normal, list):
                normal_list.append(normal[0])
            if isinstance(pose, list):
                pose_list.append(pose[0])
        if pose_list and save_pose:
            print(f"pose counts is {len(pose_list)},Save pose as *.npy files in comfyUI output....")
            for i,img in enumerate(pose_list):
                np.save(os.path.join(folder_paths.get_output_directory(),f"{i}"),np.array(img))
        seg_img = load_images(seg_list) if seg_list else zero_tensor
        normal_img = load_images(normal_list) if normal_list else zero_tensor
        depth_img = load_images(depth_list) if depth_list else zero_tensor
        pose_img = load_images(pose_list) if pose_list else zero_tensor
        mask = torch.cat(mask_list, dim=0) if mask_list else torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        
        if not torch.backends.mps.is_available():
            if torch.cuda.is_available():
                model.enable_model_cpu_offload()
        print(f"ALL inference took: {time.perf_counter() - start:.4f} seconds")
        return (seg_img, depth_img, normal_img, pose_img, mask)


NODE_CLASS_MAPPINGS = {
    "SapiensLoader": SapiensLoader,
    "SapiensSampler": SapiensSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SapiensLoader": "SapiensLoader",
    "SapiensSampler": "SapiensSampler"
}
