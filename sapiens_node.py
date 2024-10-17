# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import logging
import os
import torch
import gc
from .Sapiens_Pytorch import SapiensPredictor, SapiensConfig
from .Sapiens_Pytorch.classes_and_palettes import GOLIATH_CLASSES_FIX, GOLIATH_CLASSES
from .utils import get_models_path,tensor2cv,load_images,tensor2pil
import folder_paths

# add checkpoints dir
weigths_current_path = os.path.join(folder_paths.models_dir, "sapiens")
if not os.path.exists(weigths_current_path):
    os.makedirs(weigths_current_path)
    
try:
   folder_paths.add_model_folder_path("sapiens", weigths_current_path, False)
except:
    try:
        folder_paths.add_model_folder_path("sapiens", weigths_current_path)
        logging.warning("old comfyUI version")
    except:
        raise "please update your comfyUI version"
weigths_seg_path= os.path.join(weigths_current_path, "seg")
if not os.path.exists(weigths_seg_path):
    os.makedirs(weigths_seg_path)
weigths_depth_path= os.path.join(weigths_current_path, "depth")
if not os.path.exists(weigths_depth_path):
    os.makedirs(weigths_depth_path)
weigths_pose_path= os.path.join(weigths_current_path, "pose")
if not os.path.exists(weigths_pose_path):
    os.makedirs(weigths_pose_path)
weigths_normal_path= os.path.join(weigths_current_path, "normal")
if not os.path.exists(weigths_normal_path):
    os.makedirs(weigths_normal_path)

class SapiensLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        ckpt_list_filter=[i for i in folder_paths.get_filename_list("sapiens") if i.endswith(".pth") or i.endswith(".pt2")]
        return {
            "required": {
                "seg_ckpt": (["none"]+ckpt_list_filter,),
                "depth_ckpt": (["none"]+ckpt_list_filter,),
                "normal_ckpt": (["none"]+ckpt_list_filter,),
                "pose_ckpt": (["none"] + ckpt_list_filter,),
                "dtype": (["float32_torch","float32", "bfloat16",],),
                "minimum_person_height": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                    "round": 0.01,
                    "display": "number",
                }),
                "remove_background": ("BOOLEAN", {"default": True},),
                "use_yolo":("BOOLEAN", {"default": False},),
                "show_pose_object": ("BOOLEAN", {"default": False},),
                "seg_pellete":("BOOLEAN", {"default": True},),
            },
        }

    RETURN_TYPES = ("MODEL_SAPIEN",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "Sapiens"

    def loader_main(self, seg_ckpt,depth_ckpt,normal_ckpt,pose_ckpt,dtype,minimum_person_height,remove_background,use_yolo,show_pose_object,seg_pellete):
        
        config = SapiensConfig()
        config.model_dir=weigths_current_path
        config.pt_type = dtype
        config.detector = True if use_yolo else None
        config.remove_bg=remove_background
        config.show_pose_object=show_pose_object
        config.use_pellete=seg_pellete
        
        if dtype=="bfloat16":
            config.dtype=torch.bfloat16
        elif dtype=="float32_torch":
            config.dtype = torch.float32
        else:
            config.dtype = torch.float32
            
        config.minimum_person_height=minimum_person_height
        config=get_models_path(seg_ckpt, depth_ckpt, normal_ckpt,pose_ckpt, config,dtype)
        
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
                "seg_select":(["none"]+list(GOLIATH_CLASSES_FIX),),
                "add_seg_index":("STRING",{"default": "",}),
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
    
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE","IMAGE","MASK")
    RETURN_NAMES = ("seg_img","depth_img","normal_img","pose_img","mask")
    FUNCTION = "sampler_main"
    CATEGORY = "Sapiens"
    
    def sampler_main(self, model,image,seg_select,add_seg_index,BG_R,BG_G,BG_B):
        RGB_BG=[BG_R,BG_G,BG_B]
        
        #select body index"
        add_seg_list=add_seg_index.split(",") if add_seg_index else []
        add_seg_list=[int(i) for i in add_seg_list if int(i)<29] if add_seg_list else []
        if seg_select!="none":
            seg_select=seg_select.split(".")[-1]
            seg_select = [list(GOLIATH_CLASSES).index(seg_select)]
            if add_seg_list:
                seg_select = seg_select + add_seg_list #[0,...,27]
            
        else:
            seg_select=[]
            
        print(f"Select seg part of {seg_select} ")
        model.select=seg_select
        
        b,_,_,_=image.size()
        if b == 1:
            zero_tensor = torch.zeros_like(image, dtype=torch.float32, device="cpu")
            if model.use_pellete:#pil
                img_in = [tensor2pil(image)]
            else:
                img_in = [tensor2cv(image.squeeze())]
        else:
            image_list =torch.chunk(image, chunks=b)
            zero_tensor = torch.zeros_like(image_list[0], dtype=torch.float32, device="cpu")
            if model.use_pellete:
                img_in = [tensor2pil(i) for i in image_list] #pil
            else:
                img_in = [tensor2cv(i.squeeze()) for i in image_list]
        seg_list=[]
        mask_list=[]
        depth_list=[]
        normal_list=[]
        pose_list=[]
        for img in img_in:
            seg, depth, normal, pose, mask = model(img,seg_select,RGB_BG)
            if isinstance(seg, list):
                seg_list.append(seg[0])
                mask_list.append(mask[0])
            if isinstance(depth, list):
                depth_list.append(depth[0])
            if isinstance(normal, list):
                normal_list.append(normal[0])
            if isinstance(pose, list):
                pose_list.append(pose[0])
        seg_img = load_images(seg_list) if seg_list else zero_tensor
        normal_img = load_images(normal_list) if normal_list else zero_tensor
        depth_img = load_images(depth_list) if depth_list else zero_tensor
        pose_img = load_images(pose_list) if pose_list else zero_tensor
        mask = torch.cat(mask_list,dim=0) if mask_list else torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (seg_img, depth_img,normal_img, pose_img, mask)
        
       

NODE_CLASS_MAPPINGS = {
    "SapiensLoader": SapiensLoader,
    "SapiensSampler":SapiensSampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SapiensLoader": "SapiensLoader",
    "SapiensSampler":"SapiensSampler"
}
