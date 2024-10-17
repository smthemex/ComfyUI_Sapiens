# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from PIL import Image, ImageOps
import numpy as np
import cv2
from .Sapiens_Pytorch import SapiensDepthType, SapiensNormalType,SapiensSegmentationType,SapiensPoseEstimationType
from .Sapiens_Pytorch.common import download_hf_model,TaskType
from comfy.utils import common_upscale,ProgressBar
import folder_paths
import logging

cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

weigths_current_path = os.path.join(folder_paths.models_dir, "sapiens")
weigths_seg_path= os.path.join(weigths_current_path, "seg")

def get_path(path_name):
    if path_name != "none":
        path = folder_paths.get_full_path("sapiens", path_name)
    else:
        path = None
    return path

def get_models_path(seg_ckpt,depth_ckpt,normal_ckpt,pose_ckpt,config,dtype):
    seg_path = get_path(seg_ckpt)
    depth_path = get_path(depth_ckpt)
    normal_path = get_path(normal_ckpt)
    pose_path = get_path(pose_ckpt)
    
    if not seg_path and not depth_path and not depth_path and not pose_ckpt:
        if len(os.listdir(weigths_seg_path)) > 1 and has_file_with_extension(weigths_seg_path):
            raise "you need choice a checkpoints!"
        else:
            logging.info(
                f"No checkpoints in {weigths_seg_path},will be auto downlaod defualt seg 1b checkpoints")
            #defult torchscript version seg
            seg_file_path=os.path.join(weigths_seg_path,SapiensSegmentationType.SEGMENTATION_1B_T.value)
            if not os.path.exists(seg_file_path):
                config.local_seg_path = download_hf_model(SapiensSegmentationType.SEGMENTATION_1B_T.value,
                                                          TaskType.SEG,
                                                          weigths_seg_path, dtype)
            else:
                config.local_seg_path= seg_file_path
            config.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B_T
            config.use_torchscript=True
            return config
    if seg_path:
        config.use_torchscript_seg = False
        if "1b" in seg_path:
            if "bfloat16" in seg_path:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B_16
            elif "torchscript" in seg_path:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B_T
                config.use_torchscript_seg = True
            else:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_1B
        elif "0.6b" in seg_path:
            if "bfloat16" in seg_path:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_06B_16
            elif "torchscript" in seg_path:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_06B_T
                config.use_torchscript_seg = True
            else:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_06B
        elif "0.3b" in seg_path:
            if "bfloat16" in seg_path:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_03B_16
            elif "torchscript" in seg_path:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_03B_T
                config.use_torchscript_seg = True
            else:
                config.segmentation_type = SapiensSegmentationType.SEGMENTATION_03B
        else:
            config.segmentation_type=SapiensSegmentationType.OFF
            seg_path=""
        config.local_seg_path = seg_path
    if depth_path:
        config.use_torchscript_depth = False
        if "1b" in depth_path:
            if "bfloat16" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_1B_16
            elif "torchscript" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_1B_T
                config.use_torchscript_depth = True
            else:
                config.depth_type = SapiensDepthType.DEPTH_1B
        elif "2b" in depth_path:
            if "bfloat16" in depth_path:
               config.depth_type = SapiensDepthType.DEPTH_2B_16
            elif "torchscript" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_2B_T
                config.use_torchscript_depth = True
            else:
                config.depth_type = SapiensDepthType.DEPTH_2B
        elif "0.6b" in depth_path:
            if "bfloat16" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_06B_16
            elif "torchscript" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_06B_T
                config.use_torchscript_depth = True
            else:
                config.depth_type = SapiensDepthType.DEPTH_06B
        elif "0.3b" in depth_path:
            if "bfloat16" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_03B_16
            elif "torchscript" in depth_path:
                config.depth_type = SapiensDepthType.DEPTH_03B_T
                config.use_torchscript_depth = True
            else:
                config.depth_type = SapiensDepthType.DEPTH_03B
        else:
            config.depth_type = SapiensDepthType.OFF
            depth_path = ""
            logging.warning("checkpotin name is not fetch 0.6b,0.3b,1b,2b,depth set off")
        config.local_depth_path = depth_path
    else:
        config.depth_type = SapiensDepthType.OFF
    if normal_path:
        config.use_torchscript_normal = False
        if "1b" in normal_path:
            if "bfloat16" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_1B_16
            elif "torchscript" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_1B_T
                config.use_torchscript_normal = True
            else:
                config.normal_type = SapiensNormalType.NORMAL_1B
        elif "2b" in normal_path:
            if "bfloat16" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_2B_16
            elif "torchscript" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_2B_T
                config.use_torchscript_normal = True
            else:
                config.normal_type = SapiensNormalType.NORMAL_2B
        elif "0.6b" in normal_path:
            if "bfloat16" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_06B_16
            elif "torchscript" in depth_path:
                config.normal_type = SapiensNormalType.NORMAL_06B_T
                config.use_torchscript_normal = True
            else:
                config.normal_type = SapiensNormalType.NORMAL_06B
        elif "0.3b" in normal_path:
            if "bfloat16" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_03B_16
            elif "torchscript" in normal_path:
                config.normal_type = SapiensNormalType.NORMAL_03B_T
                config.use_torchscript_normal = True
            else:
                config.normal_type = SapiensNormalType.NORMAL_03B
        else:
            config.normal_type = SapiensNormalType.OFF
            normal_path = ""
            logging.warning("checkpotin name is not fetch 0.6b,0.3b,1b,2b,depth set off")
        config.local_normal_path = normal_path
    else:
        config.normal_type = SapiensNormalType.OFF
    if pose_path:
        config.use_torchscript_pose=False
        if "1b" in pose_path:
            if "bfloat16" in pose_path:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_1B_16
            elif "torchscript" in pose_path:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_1B_T
                config.use_torchscript_pose = True
            else:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_1B
        elif "0.6b" in pose_path:
            if "bfloat16" in pose_path:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_06B_16
            elif "torchscript" in pose_path:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_06B_T
                config.use_torchscript_pose = True
            else:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_06B
        elif "0.3b" in pose_path:
            if "bfloat16" in pose_path:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_03B_16
            elif "torchscript" in pose_path:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_03B_T
                config.use_torchscript_pose = True
            else:
                config.pose_type = SapiensPoseEstimationType.POSE_ESTIMATION_03B
        else:
            config.pose_type = SapiensPoseEstimationType.OFF
            pose_path = ""
            logging.warning("checkpotin name is not fetch 0.6b,0.3b,1b,2b,depth set off")
        config.local_pose_path = pose_path
    else:
        config.pose_type = SapiensPoseEstimationType.OFF
    return config

def has_file_with_extension(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".pth"):
            return True
        if filename.endswith(".pt2"):
            return True
    return False

def tensor2cv(tensor_image):
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu().detach()
    tensor_image=tensor_image.numpy()
    #反归一化
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def cvargb2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def cv2tensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256

def images_generator(img_list: list,):
    #get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_,Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_,np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise "unsupport image list,must be pil or cv2!!!"
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]
    
    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in=img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in,np.ndarray):
            i=cv2.cvtColor(img_in,cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            print(i.shape)
            return i
        else:
           raise "unsupport image list,must be pil,cv2 or tensor!!!"
        
    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image

def load_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry