# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import time
from tqdm import tqdm
import numpy as np
import torch
from .Sapiens_Pytorch import SapiensPredictor, SapiensConfig
from .Sapiens_Pytorch.classes_and_palettes import GOLIATH_CLASSES_FIX, GOLIATH_CLASSES
from .sapiens.predictor import Sapiens2Predictor
from .utils import get_models_path, tensor2cv, load_images, tensor2pil
from comfy_api.latest import io
import folder_paths
from .Sapiens_Pytorch.detector import Detector

node_cr_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device(
    "mps") if torch.backends.mps.is_available() else torch.device(
    "cpu")

# add checkpoints dir
weigths_sapiens_current_path = os.path.join(folder_paths.models_dir, "sapiens")
if not os.path.exists(weigths_sapiens_current_path):
    os.makedirs(weigths_sapiens_current_path)
folder_paths.add_model_folder_path("sapiens", weigths_sapiens_current_path) #  sapiens dir

for path in ["seg", "depth", "pose", "normal"]:
    weigths_current_path = os.path.join(weigths_sapiens_current_path, path)
    if not os.path.exists(weigths_current_path):
        os.makedirs(weigths_current_path)

class SapiensLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SapiensLoader",
            display_name="SapiensLoader",
            category="Sapiens",
            inputs=[
                io.Combo.Input("seg",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "seg" in i ] ),
                io.Combo.Input("depth",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "depth" in i ] ),
                io.Combo.Input("normal",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens")  if "normal" in i ] ),
                io.Combo.Input("pose",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "pose" in i ] ),
                io.Combo.Input("pointmap",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "pointmap" in i ] ),
                io.Combo.Input("albedo",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "albedo" in i ] ),
                io.Combo.Input("dtype",options= ["float32_torch", "bf16_torch", "float32", "bfloat16", ]),
                io.Float.Input("mini_person_h", default=0.5, min=0.0, max=1.0,step=0.05,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("remove_bg", default=False),
                io.Boolean.Input("use_yolo", default=False),
                io.Boolean.Input("show_pose_object", default=False),
                io.Boolean.Input("seg_pellete", default=True),
                io.Boolean.Input("convert_torchscript", default=False),
                io.Boolean.Input("V2", default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, seg,depth,normal,pose,pointmap,albedo,dtype,mini_person_h,remove_bg,use_yolo,show_pose_object,seg_pellete,convert_torchscript,V2) -> io.NodeOutput:
        if not V2 : # v1 version
            config = SapiensConfig()
            config.model_dir = weigths_current_path
            config.pt_type = dtype
            config.detector = True if use_yolo else None
            config.remove_bg = remove_bg
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
            
            config.minimum_person_height = mini_person_h
            config = get_models_path(seg, depth, normal, pose, config, dtype)
            
            # currently only for TorchScript models, convert selected FP32 TorchScript Sapiens models to BF16, save them and use them
            if convert_torchscript:
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
        else:
            # Build the model and load a pretrained checkpoint
            # model = Sapiens2(arch="sapiens2_1b", img_size=(1024, 768), patch_size=16).eval().cuda()  # img_size is (H, W)
            # model.load_state_dict(load_file(pretrained_pt))
            seg_path = folder_paths.get_full_path("sapiens",seg) if seg != "none" else None
            pointmap_path = folder_paths.get_full_path("sapiens",pointmap) if pointmap != "none" else None
            normal_path = folder_paths.get_full_path("sapiens",normal) if normal != "none" else None
            albedo_path = folder_paths.get_full_path("sapiens",albedo) if albedo != "none" else None
            pose_path = folder_paths.get_full_path("sapiens",pose) if pose != "none" else None
            pose_detector=Detector() if pose_path is not None else None
            if pointmap_path is not None and seg_path is None:
                raise("pointmap need segment model to get mask,please select a segment model")
            model=Sapiens2Predictor(seg_path,pointmap_path,albedo_path,normal_path,pose_path,pose_detector,node_cr_path,device)
            model.load_model(torch.device("cpu")) # keep model on cpu
            model.remove_bg=remove_bg
        return io.NodeOutput(model)



class SapiensSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SapiensSampler",
            display_name="SapiensSampler",
            category="Sapiens",
            inputs=[
                io.Model.Input("model"),
                io.Image.Input("image"),
                io.Combo.Input("seg_select",options= ["none"] +list(GOLIATH_CLASSES_FIX) ),
                io.String.Input("add_seg_index", default="", multiline=False), 
                io.Boolean.Input("save_pose", default=True),
                io.Int.Input("BG_R", default=255, min=0, max=255,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("BG_G", default=255, min=0, max=255,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("BG_B", default=255, min=0, max=255,step=1,display_mode=io.NumberDisplay.number),
            ], 
            outputs=[
                io.Image.Output(display_name="seg_img"),
                io.Image.Output(display_name="pose_img"),
                io.Image.Output(display_name="depth_img"),  
                io.Image.Output(display_name="normal_img"),
                io.Image.Output(display_name="pointmap_img"),
                io.Image.Output(display_name="albedo_img"),
                io.Mask.Output(display_name="mask"),
            ],
        )
    @classmethod
    def execute(cls, model, image, seg_select, add_seg_index,save_pose, BG_R, BG_G, BG_B) -> io.NodeOutput:
        start = time.perf_counter()
        zero_tensor = torch.zeros_like(image, dtype=torch.float32, device="cpu")
        b, _, _, _ = image.size()
       
        if isinstance(model, SapiensPredictor):
            if b == 1:
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
            albedo_img = zero_tensor
            pointmap_img = zero_tensor
            if not torch.backends.mps.is_available():
                if torch.cuda.is_available():
                    model.enable_model_cpu_offload()
        else:
            image_list=[image] if b == 1 else torch.chunk(image, chunks=b)
            img_in = [tensor2cv(i.squeeze()) for i in image_list]
            seg_img,normal_img,pointmap_img,albedo_img,pose_img,mask_list=model.predict(img_in)
            #print(f"seg_img shape is {seg_img[0].shape}") #seg_img shape is torch.Size([1, 1024, 1024, 3])
            mask= torch.zeros((64, 64), dtype=torch.float32, device="cpu") if mask_list is None else torch.stack([torch.from_numpy(i).float() for i in mask_list], dim=0)
            depth_img = zero_tensor
            pose_img=zero_tensor if pose_img is None else torch.cat(pose_img, dim=0)
            seg_img=zero_tensor if seg_img is None else torch.cat(seg_img, dim=0)
            normal_img=zero_tensor if normal_img is None else torch.cat(normal_img, dim=0)
            pointmap_img=zero_tensor if pointmap_img is None else torch.cat(pointmap_img, dim=0)
        print(f"ALL inference took: {time.perf_counter() - start:.4f} seconds")
        return io.NodeOutput(seg_img,pose_img, depth_img, normal_img,pointmap_img,albedo_img, mask)
