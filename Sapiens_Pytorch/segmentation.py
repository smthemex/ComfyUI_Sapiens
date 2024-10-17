import time
from enum import Enum

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import create_preprocessor, TaskType, download_hf_model,ModelManager,ImageProcessor


class SapiensSegmentationType(Enum):
    SEGMENTATION_03B = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194.pth"
    SEGMENTATION_06B = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178.pth"
    SEGMENTATION_1B = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth"
    SEGMENTATION_1B_16 = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_bfloat16.pt2"
    SEGMENTATION_06B_16 = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_bfloat16.pt2"
    SEGMENTATION_03B_16 = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_bfloat16.pt2"
    SEGMENTATION_1B_T = "sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2"
    SEGMENTATION_06B_T = "sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2"
    SEGMENTATION_03B_T = "sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2"
    OFF="off"


random = np.random.RandomState(11)
classes = ["Background", "Apparel", "Face Neck", "Hair", "Left Foot", "Left Hand", "Left Lower Arm", "Left Lower Leg",
           "Left Shoe", "Left Sock", "Left Upper Arm", "Left Upper Leg", "Lower Clothing", "Right Foot", "Right Hand",
           "Right Lower Arm", "Right Lower Leg", "Right Shoe", "Right Sock", "Right Upper Arm", "Right Upper Leg",
           "Torso", "Upper Clothing", "Lower Lip", "Upper Lip", "Lower Teeth", "Upper Teeth", "Tongue"]

colors = random.randint(0, 255, (len(classes) - 1, 3))
colors = np.vstack((np.array([0, 0, 0]), colors)).astype(np.uint8)  # Add background color
#colors = np.vstack((np.array([128, 128, 128]), colors)).astype(np.uint8)  # Add background color
colors = colors[:, ::-1]


def draw_segmentation_map(segmentation_map: np.ndarray) -> np.ndarray:
    h, w = segmentation_map.shape
    
    segmentation_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        segmentation_img[segmentation_map == i] = color

    return segmentation_img


def postprocess_segmentation(results: torch.Tensor, img_shape: tuple[int, int]) -> np.ndarray:
    result = results[0].cpu()

    # Upsample the result to the original image size
    logits = F.interpolate(result.unsqueeze(0), size=img_shape, mode="bilinear").squeeze(0)

    # Perform argmax to get the segmentation map
    segmentation_map = logits.argmax(dim=0, keepdim=True)

    # Covert to numpy array
    segmentation_map = segmentation_map.float().numpy().squeeze()

    return segmentation_map


class SapiensSegmentation():
    def __init__(self,
                 type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B,loacl_seg="",pt_type="float32_torch",model_dir="",img_size=(1024, 768),use_torchscript=True,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        self.loacl_seg = loacl_seg
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.img_size=img_size
        self.use_torchscript=use_torchscript
        if self.loacl_seg:
            path=self.loacl_seg
        else:
            path = download_hf_model(type.value, TaskType.SEG,self.model_dir,dtype=self.pt_type)
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
        self.preprocessor = create_preprocessor(input_size=self.img_size)  # Only these values (1024, 768) seem to work well

    def __call__(self, img: np.ndarray) -> np.ndarray:
        start = time.perf_counter()

        # Model expects BGR, but we change to RGB here because the preprocessor will switch the channels also
        input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.preprocessor(input).to(self.device).to(self.dtype)

        with torch.inference_mode():
            results = self.model(tensor)
        segmentation_map = postprocess_segmentation(results, img.shape[:2])

        print(f"Segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return segmentation_map

class SapiensSeg():
    def __init__(self,
                 type: SapiensSegmentationType = SapiensSegmentationType.SEGMENTATION_1B,local_seg="",pt_type="float32_torch",model_dir="",use_torchscript=True,
                 dtype: torch.dtype = torch.float32):
        self.local_seg = local_seg
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.use_torchscript=use_torchscript
        self.model_name=type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.image_processor = ImageProcessor()
        if self.local_seg:
            path = self.local_seg
        else:
            path = download_hf_model(self.model_name.value, TaskType.SEG, self.model_dir, dtype=self.pt_type)
        self.model = ModelManager.load_model(path,self.device)
    
    def __call__(self, image,select_obj,RGB_BG) :
        start = time.perf_counter()
        with torch.inference_mode():
            result, mask,seg_pred = self.image_processor.process_image(image, self.model,select_obj,RGB_BG)
        print(f"Segmentation inference took: {time.perf_counter() - start:.4f} seconds")
        return result, mask,seg_pred


