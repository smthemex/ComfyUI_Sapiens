# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from enum import Enum
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor, TaskType, download_hf_model,ImageProcessorNormal,ModelManager


class SapiensNormalType(Enum):
    OFF = "off"
    NORMAL_03B = "sapiens_0.3b_normal_render_people_epoch_66.pth"
    NORMAL_06B = "sapiens_0.6b_normal_render_people_epoch_200.pth"
    NORMAL_1B = "sapiens_1b_normal_render_people_epoch_115.pth"
    NORMAL_2B = "sapiens_2b_normal_render_people_epoch_70.pth"
    NORMAL_03B_T = "sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2"
    NORMAL_06B_T = "sapiens_0.6b_normal_render_people_epoch_200_torchscript.pt2"
    NORMAL_1B_T = "sapiens_1b_normal_render_people_epoch_115_torchscript.pt2"
    NORMAL_2B_T = "sapiens_2b_normal_render_people_epoch_70_torchscript.pt2"
    NORMAL_03B_16 = "sapiens_0.3b_normal_render_people_epoch_66_bfloat16.pt2"
    NORMAL_06B_16 = "sapiens_0.6b_normal_render_people_epoch_200_bfloat16.pt2"
    NORMAL_1B_16 = "sapiens_1b_normal_render_people_epoch_115_bfloat16.pt2"
    NORMAL_2B_16 = "sapiens_2b_normal_render_people_epoch_70_bfloat16.pt2"


def draw_normal_map(normal_map: np.ndarray) -> np.ndarray:
    # Normalize the normal map
    normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small epsilon to avoid division by zero
    normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)

    # Convert to BGR
    return cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR)


def postprocess_normal(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].detach().cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Covert to numpy array
    normal_map = logits.float().numpy().transpose(1, 2, 0)

    return normal_map


class SapiensNormal():
    def __init__(self,
                 type: SapiensNormalType = SapiensNormalType.NORMAL_03B,local_normal="",pt_type="float32_torch",model_dir="",img_size=(1024, 768),use_torchscript=True,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        self.local_normal=local_normal
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.img_size = img_size
        self.use_torchscript = use_torchscript
        self.device = device
        self.dtype = dtype
        if self.local_normal:
            path= self.local_normal
        else:
            path = download_hf_model(type.value, TaskType.NORMAL,self.model_dir,dtype=self.pt_type)
        if self.use_torchscript:
            model = torch.jit.load(path)
            model = model.eval()
            model.to(device).to(dtype)
        else:
            model = torch.export.load(path).module()
            dtype = torch.half if self.dtype=="float16" else torch.bfloat16
            model.to(dtype)
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
        self.model = model
        self.preprocessor = create_preprocessor(input_size=self.img_size)  # Only these values seem to work well(1024, 768)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)

        normals = postprocess_normal(results, img.shape[:2])
        #print(f"Normal inference took: {time.perf_counter() - start:.4f} seconds")
        return normals
    
    def enable_model_cpu_offload(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()
    def move_to_cuda(self):
        self.model.to("cuda")

class NormalSapiens():
    def __init__(self,
                 type: SapiensNormalType = SapiensNormalType.NORMAL_03B,local_normal="",
                 pt_type="float32_torch", model_dir="", use_torchscript=True,
                 dtype: torch.dtype = torch.float32):
        self.local_normal = local_normal
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.use_torchscript = use_torchscript
        self.model_name = type
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.dtype = dtype
        self.image_processor = ImageProcessorNormal()
        if self.local_normal:
            path = self.local_normal
        else:
            path = download_hf_model(self.model_name.value, TaskType.NORMAL, self.model_dir, dtype=self.pt_type)
        self.model = ModelManager.load_model(path, self.device)
    
    def __call__(self, image, if_seg, seg_in,select_obj,RGB_BG):
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.image_processor.process_image(image, self.model, if_seg, seg_in,select_obj,RGB_BG, model_dtype=self.dtype)
        #print(f"Normal inference took: {time.perf_counter() - start:.4f} seconds")
        return result
    
    def enable_model_cpu_offload(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()
    
    def move_to_cuda(self):
        self.model.to("cuda")
