import os
from typing import List

import cv2
import requests
from tqdm import tqdm
from enum import Enum
from huggingface_hub import hf_hub_download
from torchvision import transforms
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from .classes_and_palettes import GOLIATH_PALETTE, GOLIATH_CLASSES

class TaskType(Enum):
    DEPTH = "depth"
    NORMAL = "normal"
    SEG = "seg"
    POSE = "pose"


class ModelManager:
    @staticmethod
    def load_model(path: str,device):
        model = torch.jit.load(path)
        model.eval()
        model.to(device)
        return model

    @staticmethod
    @torch.inference_mode()
    def run_model(model, input_tensor, height, width):
        output = model(input_tensor)
        output = F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
        _, preds = torch.max(output, 1)
        return preds
    
    @staticmethod
    @torch.inference_mode()
    def run_model_depth(model, input_tensor, height, width): #depth,normal
        output = model(input_tensor)
        return F.interpolate(output, size=(height, width), mode="bilinear", align_corners=False)
    
    
class ImageProcessor:
    def __init__(self):
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5 / 255, 116.5 / 255, 103.5 / 255],
                                 std=[58.5 / 255, 57.0 / 255, 57.5 / 255]),
        ])
    
    def process_image(self, image: Image.Image, model,select_obj,RGB_BG):
        input_tensor = self.transform_fn(image).unsqueeze(0).to("cuda")
        preds = ModelManager.run_model(model, input_tensor, image.height, image.width)
        mask = preds.squeeze(0).cpu().numpy()
        
        # Visualize the segmentation
        blended_image,blended_mask = self.visualize_pred_with_overlay(image, mask,select_obj,RGB_BG)
        
        return blended_image, blended_mask,preds
    
    @staticmethod
    def visualize_pred_with_overlay(img, sem_seg, select_obj,RGB_BG):
        img_np = np.array(img.convert("RGB"))
        sem_seg = np.array(sem_seg)
        
        num_classes = len(GOLIATH_CLASSES)
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        overlay = np.zeros((*sem_seg.shape, 3), dtype=np.uint8)
        empty_np=np.zeros_like(img_np.shape)
        if select_obj:
            labels_s=np.array(select_obj,dtype=np.int64)
            colors_s=[GOLIATH_PALETTE[label] for label in labels_s]
            for label, color in zip(labels_s, colors_s):
                overlay[sem_seg == label, :] = color
        else:
            labels = np.array(ids, dtype=np.int64)
            labels=labels[labels != 0] #remove bg
            colors = [GOLIATH_PALETTE[label] for label in labels]
            for label, color in zip(labels, colors):
                overlay[sem_seg == label, :] = color
        x = np.uint8(empty_np + overlay)
        seg_gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        ret, mask_ = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY)
        img_np[mask_ != 255] =RGB_BG #[255, 255, 255]
        blended_seg = np.uint8(img_np)
        blended_mask = get_mask(x.copy())
        return Image.fromarray(blended_seg),blended_mask


class ImageProcessorDepth:
    def __init__(self):
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5 / 255, 116.5 / 255, 103.5 / 255],
                                 std=[58.5 / 255, 57.0 / 255, 57.5 / 255]),
        ])
    

    def process_image(self, image: Image.Image, depth_model, if_seg,seg_in,RGB_BG):

        input_tensor = self.transform_fn(image).unsqueeze(0).to("cuda")
        depth_output = ModelManager.run_model_depth(depth_model, input_tensor, image.height, image.width)
        depth_map = depth_output.squeeze().cpu().numpy()
        
        if if_seg:
            seg_mask = (seg_in.argmax(dim=1) > 0).float().cpu().numpy()[0]
            depth_map[seg_mask == 0] = RGB_BG
            #depth_map[seg_mask == 0] = np.nan
        
        depth_colored = self.colorize_depth_map(depth_map)
       
        return Image.fromarray(depth_colored)
    
    @staticmethod
    def colorize_depth_map(depth_map):
        depth_foreground = depth_map[~np.isnan(depth_map)]
        if len(depth_foreground) > 0:
            min_val, max_val = np.nanmin(depth_foreground), np.nanmax(depth_foreground)
            depth_normalized = (depth_map - min_val) / (max_val - min_val)
            depth_normalized = 1 - depth_normalized
            depth_normalized = np.nan_to_num(depth_normalized, nan=0)
            cmap = plt.get_cmap('inferno')
            depth_colored = (cmap(depth_normalized) * 255).astype(np.uint8)[:, :, :3]
        else:
            depth_colored = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)
        return depth_colored


class ImageProcessorNormal:
    def __init__(self):
        self.transform_fn = transforms.Compose([
            transforms.Resize((1024, 768)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.5/255, 116.5/255, 103.5/255], std=[58.5/255, 57.0/255, 57.5/255]),
        ])

    def process_image(self, image: Image.Image, normal_model, if_seg,seg_in,RGB_BG):
        # Load models here instead of storing them as class attributes
        input_tensor = self.transform_fn(image).unsqueeze(0).to("cuda")

        # Run normal estimation
        normal_output = ModelManager.run_model(normal_model, input_tensor, image.height, image.width)
        normal_map = normal_output.squeeze().cpu().numpy().transpose(1, 2, 0)

        # Create a copy of the normal map for visualization
        normal_map_vis = normal_map.copy()

        # Run segmentation
        if if_seg:
            seg_mask = (seg_in.argmax(dim=1) > 0).float().cpu().numpy()[0]

            # Apply segmentation mask to normal maps
            normal_map[seg_mask == 0] = np.nan  # Set background to NaN for NPY file
            #normal_map_vis[seg_mask == 0] = -1  # Set background to -1 for visualization
            normal_map_vis[seg_mask == 0] = RGB_BG  # Set background to -1 for visualization

        # Normalize and visualize normal map
        normal_map_vis = self.visualize_normal_map(normal_map_vis)
        return Image.fromarray(normal_map_vis)

    @staticmethod
    def visualize_normal_map(normal_map):
        normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
        normal_map_normalized = normal_map / (normal_map_norm + 1e-5)
        normal_map_vis = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
        return normal_map_vis



def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(model_name: str, task_type: TaskType, model_dir: str = 'models',dtype="float32"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    path = model_dir + f"/{task_type.value}/" + model_name
    if os.path.exists(path):
        return path

    print(f"Model {model_name} not found, downloading from Hugging Face Hub...")
    
    if "0.3b" in model_name:
        model_version="0.3b"
    elif "0.6b" in model_name:
        model_version = "0.6b"
    elif "1b" in model_name:
        model_version = "1b"
    elif "2b" in model_name:
        model_version = "2b"
    else:
        raise "get unsupport model_name"

    if dtype == "float32":
        repo_id=f"facebook/sapiens-{task_type.value}-{model_version}"
    elif  dtype == "float32_torch":
        repo_id = f"facebook/sapiens-{task_type.value}-{model_version}-torchscript" #torchscript
    else:
        repo_id = f"facebook/sapiens-{task_type.value}-{model_version}-{dtype}"#bfloat16
        
    real_dir=os.path.join(model_dir,f"{task_type.value}")
    
    hf_hub_download(repo_id=repo_id, filename=model_name, local_dir=real_dir)
    print("Model downloaded successfully to", path)
    return path


def create_preprocessor(input_size: tuple[int, int],
                        mean: List[float] = (0.485, 0.456, 0.406),
                        std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               transforms.Lambda(lambda x: x.unsqueeze(0))
                               ])


def pose_estimation_preprocessor(input_size: tuple[int, int],
                                 mean: List[float] = (0.485, 0.456, 0.406),
                                 std: List[float] = (0.229, 0.224, 0.225)):
    return transforms.Compose([transforms.ToPILImage(),
                               transforms.Resize(input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=mean, std=std),
                               ])

def get_mask(draw):
    seg_gray = cv2.cvtColor(draw.copy(), cv2.COLOR_BGR2GRAY)
    ret, mask_ = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY)
    mask_image = Image.fromarray(mask_).convert('RGBA')
    alpha = mask_image.split()[0]
    bg = Image.new("L", mask_image.size)
    mask_image = Image.merge('RGBA', (bg, bg, bg, alpha))
    mask_image = torch.from_numpy(np.array(mask_image).astype(np.float32) / 255.0).unsqueeze(0)
    real_mask = torch.tensor([mask_image[0, :, :, 3].tolist()])
    return real_mask