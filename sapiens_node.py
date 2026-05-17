# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import time
import numpy as np
import torch
from .Sapiens_Pytorch import SapiensPredictor, SapiensConfig
from .Sapiens_Pytorch.classes_and_palettes import GOLIATH_CLASSES_FIX
from .sapiens.predictor import Sapiens2Predictor
from .utils import get_models_path, tensor2cv,  tensor2pil
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


class SapiensLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SapiensLoader",
            display_name="SapiensLoader",
            category="Sapiens",
            inputs=[
                io.Combo.Input("seg",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "seg" in i or "mIoU" in i ] ),
                io.Combo.Input("depth",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "depth" in i or ("render_people" in i and "normal" not in i) ] ),
                io.Combo.Input("normal",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens")  if "normal" in i ] ),
                io.Combo.Input("pose",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "pose" in i or "goliath_AP" in i ] ),
                io.Combo.Input("pointmap",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "pointmap" in i ] ),
                io.Combo.Input("albedo",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "albedo" in i ] ),
                io.Combo.Input("matting",options= ["none"] + [i for i in folder_paths.get_filename_list("sapiens") if "matting" in i ] ),
                io.Combo.Input("dtype",options= [ "float32", "bfloat16",]),
                io.Float.Input("mini_person_h", default=0.5, min=0.0, max=1.0,step=0.05,display_mode=io.NumberDisplay.number),
                io.Boolean.Input("show_pose_object", default=False),
                io.Boolean.Input("convert_torchscript", default=False),
                io.Boolean.Input("V2", default=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                ],
            )
    @classmethod
    def execute(cls, seg,depth,normal,pose,pointmap,albedo,matting,dtype,mini_person_h,show_pose_object,convert_torchscript,V2) -> io.NodeOutput:
        if dtype == "bfloat16":
            infer_dtype = torch.bfloat16
        else:
            infer_dtype = torch.float32
        if not V2 : # v1 version
            config = SapiensConfig()
            config.model_dir = weigths_sapiens_current_path
            config.pt_type = dtype
            config.show_pose_object = show_pose_object
            config.dtype = infer_dtype   
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
            seg_path = folder_paths.get_full_path("sapiens",seg) if seg != "none" else None
            pointmap_path = folder_paths.get_full_path("sapiens",pointmap) if pointmap != "none" else None
            normal_path = folder_paths.get_full_path("sapiens",normal) if normal != "none" else None
            albedo_path = folder_paths.get_full_path("sapiens",albedo) if albedo != "none" else None
            pose_path = folder_paths.get_full_path("sapiens",pose) if pose != "none" else None
            pose_detector=Detector() if pose_path is not None else None
            matting_path = folder_paths.get_full_path("sapiens",matting) if matting != "none" else None
            if pointmap_path is not None and seg_path is None:
                raise("pointmap need segment model to get mask,please select a segment model")
            model=Sapiens2Predictor(seg_path,pointmap_path,albedo_path,normal_path,pose_path,matting_path,pose_detector,node_cr_path,device,infer_dtype)
            model.load_model(torch.device("cpu")) # keep model on cpu
            model.remove_bg=True
            model.show_pose_object=show_pose_object
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
                io.Boolean.Input("save_pose", default=False),
                io.Int.Input("BG_R", default=255, min=0, max=255,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("BG_G", default=255, min=0, max=255,step=1,display_mode=io.NumberDisplay.number),
                io.Int.Input("BG_B", default=255, min=0, max=255,step=1,display_mode=io.NumberDisplay.number),
                io.Conditioning.Input("cond",optional=True),
            ], 
            outputs=[
                io.Image.Output(display_name="seg_img"),
                io.Image.Output(display_name="pose_img"),
                io.Image.Output(display_name="depth_img"),  
                io.Image.Output(display_name="normal_img"),
                io.Image.Output(display_name="pointmap_img"),
                io.Image.Output(display_name="albedo_img"),
                io.Image.Output(display_name="matting_img"),
                io.Mask.Output(display_name="mask"),
                
            ],
        )
    @classmethod
    def execute(cls, model, image,save_pose, BG_R, BG_G, BG_B, cond=None,) -> io.NodeOutput:
        start = time.perf_counter()
        zero_tensor = torch.zeros_like(image, dtype=torch.float32, device="cpu")
        image_list=[image] if image.size()[0] == 1 else torch.chunk(image, chunks=image.size()[0])
        albedo_list,pointmap_list,depth_list,matting_list=[],[],[],[]
        RGB_BG = [BG_R, BG_G, BG_B]
        cond=[] if not cond else cond
        print("start predict...")
        if isinstance(model, SapiensPredictor):
            if not torch.backends.mps.is_available():
                if torch.cuda.is_available():
                    model.move_to_cuda()  
            
            img_in=[tensor2pil(i) for i in image_list]
            seg_list, depth_list, normal_list, pose_list, mask_list = model(img_in, cond, RGB_BG)
            
            mask = torch.cat(mask_list, dim=0) if mask_list else torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            if not torch.backends.mps.is_available():
                if torch.cuda.is_available():
                    model.enable_model_cpu_offload()
        else:
            img_in = [tensor2cv(i.squeeze()) for i in image_list]
            seg_list,normal_list,pointmap_list,albedo_list,pose_list,matting_list,mask_list,matting_mask=model.predict(img_in,cond, RGB_BG)
            mask_list=matting_mask if len(matting_mask)>0 else mask_list
            mask= torch.zeros((64, 64), dtype=torch.float32, device="cpu") if not mask_list  else torch.stack([torch.from_numpy(i).float() for i in mask_list], dim=0)

        depth_img = zero_tensor if not depth_list  else torch.cat(depth_list, dim=0)
        pose_img=zero_tensor if not pose_list else torch.cat(pose_list, dim=0)
        seg_img=zero_tensor if not seg_list  else torch.cat(seg_list, dim=0)
        normal_img=zero_tensor if not normal_list  else torch.cat(normal_list, dim=0)
        pointmap_img=zero_tensor if  not pointmap_list  else torch.cat(pointmap_list, dim=0)
        albedo_img=zero_tensor if not albedo_list  else torch.cat(albedo_list, dim=0)
        matting_image=zero_tensor if not matting_list  else torch.cat(matting_list, dim=0)

        if pose_list and save_pose:
            print(f"pose counts is {len(pose_list)},Save pose as *.npy files in comfyUI output....")
            for i,img in enumerate(pose_list):
                np.save(os.path.join(folder_paths.get_output_directory(),f"{i}"),tensor2cv(img))
        print(f"ALL inference took: {time.perf_counter() - start:.4f} seconds")
        return io.NodeOutput(seg_img,pose_img, depth_img, normal_img,pointmap_img,albedo_img,matting_image, mask )

class SapiensSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SapiensSplit",
            display_name="SapiensSplit",
            category="Sapiens",
            inputs=[
                io.Boolean.Input("keep_all", default=False),
                io.Boolean.Input("Background", default=False),
                io.Boolean.Input("Apparel", default=False),
                io.Boolean.Input("Eyeglass", default=False),
                io.Boolean.Input("Face_Neck", default=False),
                io.Boolean.Input("Hair", default=False),
                io.Boolean.Input("Left_Foot", default=False),
                io.Boolean.Input("Left_Hand", default=False),
                io.Boolean.Input("Left_Lower_Arm", default=False),
                io.Boolean.Input("Left_Lower_Leg", default=False),
                io.Boolean.Input("Left_Shoe", default=False),
                io.Boolean.Input("Left_Sock", default=False),
                io.Boolean.Input("Left_Upper_Arm", default=False),
                io.Boolean.Input("Left_Upper_Leg", default=False),
                io.Boolean.Input("Lower_Clothing", default=False),
                io.Boolean.Input("Right_Foot", default=False),
                io.Boolean.Input("Right_Hand", default=False),
                io.Boolean.Input("Right_Lower_Arm", default=False),
                io.Boolean.Input("Right_Lower_Leg", default=False),
                io.Boolean.Input("Right_Shoe", default=False),
                io.Boolean.Input("Right_Sock", default=False),
                io.Boolean.Input("Right_Upper_Arm", default=False),
                io.Boolean.Input("Right_Upper_Leg", default=False),
                io.Boolean.Input("Torso", default=False),
                io.Boolean.Input("Upper_Clothing", default=False),
                io.Boolean.Input("Lower_Lip", default=False),
                io.Boolean.Input("Upper_Lip", default=False),
                io.Boolean.Input("Lower_Teeth", default=False),
                io.Boolean.Input("Upper_Teeth", default=False),
                io.Boolean.Input("Tongue", default=False),
            ], 
            outputs=[
                io.Conditioning.Output(display_name="conds"),
            ],
        )
    @classmethod
    def execute(cls, keep_all,**kwargs) -> io.NodeOutput:
        option_names = [
            "Background", "Apparel", "Eyeglass","Face_Neck", "Hair",
            "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe",
            "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing", "Right_Foot",
            "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock",
            "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip",
            "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"
            ]
        selected_indices = []
        if not keep_all:
            for i, name in enumerate(option_names):
                if kwargs.get(name, False):  
                    selected_indices.append(i)
        # select body index
        seg_select_all = [list(GOLIATH_CLASSES_FIX)[i] for i in selected_indices] if selected_indices else "seg default hunman map."
        print(f"Select seg part of {seg_select_all},index is {selected_indices} ")
        return io.NodeOutput(selected_indices)
