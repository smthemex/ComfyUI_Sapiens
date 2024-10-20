from enum import Enum
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor, download_hf_model, TaskType,ImageProcessorDepth,ModelManager


class SapiensDepthType(Enum):
    OFF = "off"
    DEPTH_03B = "sapiens_0.3b_render_people_epoch_100_torchscript.pt2"
    DEPTH_06B = "sapiens_0.6b_render_people_epoch_70_torchscript.pt2"
    DEPTH_1B = "sapiens_1b_render_people_epoch_88_torchscript.pt2"
    DEPTH_2B = "sapiens_2b_render_people_epoch_25_torchscript.pt2"
    DEPTH_03B_T = "sapiens_0.3b_render_people_epoch_100_torchscript.pt2"
    DEPTH_06B_T = "sapiens_0.6b_render_people_epoch_70_torchscript.pt2"
    DEPTH_1B_T = "sapiens_1b_render_people_epoch_88_torchscript.pt2"
    DEPTH_2B_T = "sapiens_2b_render_people_epoch_25_torchscript.pt2"
    DEPTH_03B_16 = "sapiens_0.3b_render_people_epoch_100_bfloat16.pt2"
    DEPTH_06B_16 = "sapiens_0.6b_render_people_epoch_70_bfloat16.pt2"
    DEPTH_1B_16 = "sapiens_1b_render_people_epoch_88_bfloat16.pt2"
    DEPTH_2B_16 = "sapiens_2b_render_people_epoch_25_bfloat16.pt2"


def draw_depth_map(depth_map: np.ndarray) -> np.ndarray:
    min_depth, max_depth = np.min(depth_map), np.max(depth_map)

    norm_depth_map = 1 - (depth_map - min_depth) / (max_depth - min_depth)
    norm_depth_map = (norm_depth_map * 255).astype(np.uint8)

    # Normalize and color the image
    color_depth = cv2.applyColorMap(norm_depth_map, cv2.COLORMAP_INFERNO)
    color_depth[depth_map == 0] = 128
    return color_depth


def postprocess_depth(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Covert to numpy array
    depth_map = logits.float().numpy().squeeze()
    return depth_map


class SapiensDepth():
    def __init__(self,
                 type: SapiensDepthType = SapiensDepthType.DEPTH_03B,local_depth="",pt_type="float32_torch",model_dir="",img_size=(1024, 768),use_torchscript=True,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        self.local_depth=local_depth
        self.model_dir=model_dir
        self.pt_type=pt_type
        self.img_size = img_size
        self.use_torchscript = use_torchscript
        if self.local_depth:
            path=self.local_depth
        else:
            path = download_hf_model(type.value, TaskType.DEPTH,self.model_dir,dtype=self.pt_type)
        self.device = device
        self.dtype = dtype
        if self.use_torchscript:
            model = torch.jit.load(path)
            model = model.eval()
            model.to(device).to(dtype)
        else:
            model = torch.export.load(path).module()
            dtype = torch.half if self.dtype == "float16" else torch.bfloat16
            model.to(dtype)
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
        self.model = model

        self.preprocessor = create_preprocessor(input_size= self.img_size)  # Only these values seem to work well 1024, 768

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)

        depth_map = postprocess_depth(results, img.shape[:2])
        print(f"Depth inference took: {time.perf_counter() - start:.4f} seconds")
        return depth_map


class DepthSapiens():
    def __init__(self,
                 type: SapiensDepthType = SapiensDepthType.DEPTH_03B,local_depth="",
                 pt_type="float32_torch", model_dir="", use_torchscript=True,
                 dtype: torch.dtype = torch.float32):
        self.local_depth = local_depth
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.use_torchscript = use_torchscript
        self.model_name = type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.image_processor = ImageProcessorDepth()
        if self.local_depth:
            path = self.local_depth
        else:
            path = download_hf_model(self.model_name.value, TaskType.DEPTH, self.model_dir, dtype=self.pt_type)
        self.model = ModelManager.load_model(path, self.device)
    
    def __call__(self, image, if_seg,seg_in,RGB_BG):
        start = time.perf_counter()
        with torch.inference_mode():
            result= self.image_processor.process_image(image, self.model,if_seg,seg_in,RGB_BG, model_dtype=self.dtype)
        print(f"Depth inference took: {time.perf_counter() - start:.4f} seconds")
        return result
