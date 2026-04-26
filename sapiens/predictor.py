
from pathlib import Path
from typing import Optional, Union
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch
from safetensors.torch import load_file
from .engine.config import Config
from .engine.datasets import Compose
from .registry import MODELS
import torch.nn.functional as F
from .dense.src.visualizers import SegVisualizer
from .pose.tools.vis.pose_render_utils import visualize_keypoints
from .pose.src.datasets import parse_pose_metainfo, UDPHeatmap


class Sapiens2Predictor:
    def __init__(self,seg_path,pointmap_path,albedo_path,normal_path,pose_path,pose_detector,node_dir,device):
        self.seg_path=seg_path
        self.pointmap_path=pointmap_path
        self.albedo_path=albedo_path
        self.normal_path=normal_path
        self.node_dir=node_dir
        self.device=device
        self.use_pellete=False
        self.remove_bg=True
        self.class_palette_type="dome29"
        self.no_black_background=False
        self.no_save_predictions=True
        self.with_normal=False
        self.no_save_json=True
        self.pose_path=pose_path
        self.pose_detector=pose_detector
    
    def load_model(self,device):
        self.load_device=device
        self.model_seg=load_model(self.seg_path,self.node_dir,device) if self.seg_path else None
        self.model_pointmap=load_model(self.pointmap_path,self.node_dir,device) if self.pointmap_path else None
        self.model_albedo=load_model(self.albedo_path,self.node_dir,device) if self.albedo_path else None
        self.model_normal=load_model(self.normal_path,self.node_dir,device) if self.normal_path else None
        self.model_pose=load_model(self.pose_path,self.node_dir,device) if self.pose_path else None

    def predict(self,image):
        seg_img,normal_img,pointmap_img,albedo_img,mask_list=None,None,None,None,None
        if self.model_seg:
            if self.load_device != self.device:# move to device
                self.model_seg.to(self.device)
            seg_img,mask_list= seg_predict(self.model_seg,image) # list ,list of labels
            self.model_seg.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        if self.model_pose:
            if self.load_device != self.device:# move to device
                self.model_pose.to(self.device)
            pose_img=pose_predict(self.model_pose,self.pose_detector,image,self.node_dir,)
            self.model_pose.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        if self.model_normal:
            if self.load_device != self.device:# move to device
                self.model_normal.to(self.device)
            normal_img=normal_predict(self.model_normal,image, mask_list)
            self.model_normal.to(torch.device("cpu"))
            torch.cuda.empty_cache()
        if self.model_pointmap:
            if self.load_device != self.device:# move to device
                self.model_pointmap.to(self.device)
            pointmap_img=pointmap_predict(self.model_pointmap, image, mask_list)
            self.model_pointmap.to(torch.device("cpu"))
        if self.model_albedo:
            if self.load_device != self.device:# move to device
                   self.model_albedo.to(self.device)
            albedo_img=albedo_predict(self.model_albedo, image, mask_list)
            self.model_albedo.to(torch.device("cpu"))
        
        return seg_img,normal_img,pointmap_img,albedo_img,pose_img,mask_list


def albedo_predict(model, images,mask_list):
    image_list = []
    if mask_list is None:
        mask_list = [None] * len(images)
    for image,mask in tqdm(zip(images,mask_list), total=len(images)):
        if mask is None:
            mask = np.ones_like(image[:, :, 0], dtype=bool)
        ##------------------------------------------
        data = model.pipeline(dict(img=image))  ## resize and pad
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs, data_samples = data["inputs"], data["data_samples"]

        ## pointmap is 1 x 3 x H x W, scale is 1 x 1
        with torch.no_grad():
            albedo = model(inputs)
            albedo = albedo.clamp(0, 1)  # clamp to [0, 1]

        # ------------------------------------------
        pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"]["padding_size"]
        albedo = albedo[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]

        albedo = F.interpolate(
            albedo,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        albedo = albedo.squeeze(0).cpu().numpy().transpose(1, 2, 0)  ## H x W x 3
        albedo = (albedo * 255).astype(np.uint8)
        albedo[mask == 0] = [100, 100, 100]

        image = torch.from_numpy(albedo.astype(np.float32) / 255.0).unsqueeze(0)
        image_list.append(image)

    return image_list




def seg_predict(model, images):
    #class_palette_type = args.class_palette_type
    visualizer = SegVisualizer(class_palette_type=model.class_palette_type, with_labels=False)
    image_list=[]
    mask_list=[]
    for i, image in tqdm(enumerate(images), total=len(images)):
        data = model.pipeline(dict(img=image))  ## resize and pad
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs = data["inputs"]

        ## pointmap is 1 x 3 x H x W, scale is 1 x 1
        with torch.no_grad():
            seg_logits = model(inputs)

        # resize prediction to image size
        seg_logits = F.interpolate(
            seg_logits, size=image.shape[:2], mode="bilinear"
        )  ## 1 x C x H x W
        pred_labels = seg_logits.argmax(dim=1).cpu().numpy()  ## 1 x H x W
        pred_labels = pred_labels.squeeze(0)  ## H x W

        vis_seg = visualizer._visualize_segmentation(image, pred_labels)

        vis_seg_rgb = cv2.cvtColor(vis_seg, cv2.COLOR_BGR2RGB) 
        image = torch.from_numpy(vis_seg_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        image_list.append(image)
        mask=(pred_labels > 0).astype(bool)
        mask_list.append(mask)
    return image_list,mask_list

def pointmap_predict(model, images,mask_list,):
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "open3d is required for pointmap visualization. "
            "Install with: pip install open3d  (or `pip install -e .[pointmap]`)"
        ) from e    
        # Get image list
    from .dense.tools.vis.vis_pointmap import load_image_and_mask, resize_pointmap, process_depth_map_with_bounds, compute_surface_normals
  

    # ============== Pass 1: inference (cached) + percentile collection ==============
    per_frame_percentiles = []  # list of (p1, p99) of foreground depth per frame
    #for image_name in tqdm(image_names, desc="pass 1: inference"):
    depth_list=[]
    for image,mask in tqdm(zip(images,mask_list), total=len(images),desc="pass 1: inference"):

        # base_path = os.path.join(args.output, image_name.rsplit(".")[0])
        #depth_npy_path = f"{base_path}_depth.npy"

        #image, mask = load_image_and_mask(input_dir, seg_dir, image_name)

        # if os.path.exists(depth_npy_path):
        #     depth = np.load(depth_npy_path).astype(np.float32)
        # else:
            # if model is None:
            #     model = init_model(args.config, args.checkpoint, device=args.device)

        data = model.pipeline(dict(img=image))  ## resize and pad
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs, data_samples = data["inputs"], data["data_samples"]

        ## pointmap is 1 x 3 x H x W, scale is 1 x 1
        with torch.no_grad():
            pointmap, scale = model(inputs)
            pointmap = pointmap / scale  ## convert pointmap to metric

        assert (
            pointmap.shape[0] == 1
            and pointmap.shape[2] == inputs.shape[2]
            and pointmap.shape[3] == inputs.shape[3]
        )

        pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"][
            "padding_size"
        ]
        pointmap = pointmap[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]

        pointmap = resize_pointmap(
            pointmap,
            target_height=mask.shape[0],
            target_width=mask.shape[1],
            smooth=True,
        )
        pointmap = pointmap.squeeze(0).cpu().numpy().transpose(1, 2, 0)  ## H x W x 3

        depth = pointmap[:, :, 2].astype(np.float32)
        #np.save(depth_npy_path, depth.astype(np.float16))
        depth_list.append(depth)
        if not model.no_save_predictions:
            points = pointmap[mask > 0].reshape(-1, 3)  ## N x 3
            pc = o3d.geometry.PointCloud()
            colors = image[mask > 0] / 255.0
            colors = colors[:, [2, 1, 0]]  # Convert BGR to RGB

            pc.points = o3d.utility.Vector3dVector(points)
            pc.colors = o3d.utility.Vector3dVector(colors)
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=0.05, resolution=20
            )
            sphere.translate([0, 0, 0])  # Center the sphere at the origin
            sphere.paint_uniform_color([0, 0, 1])  # Color the sphere blue
            sphere_pc = sphere.sample_points_poisson_disk(number_of_points=500)
            sphere_pc.colors = o3d.utility.Vector3dVector(
                [[0, 0, 1] for _ in range(len(sphere_pc.points))]
            )
            pc = pc + sphere_pc
            o3d.io.write_point_cloud("sapiens.ply", pc)

        depth_fg = depth[mask > 0]
        if len(depth_fg) > 0:
            p1, p99 = np.percentile(depth_fg, [1, 99])
            per_frame_percentiles.append((float(p1), float(p99)))

    # ============== Aggregate global bounds: min(p1) and max(p99) ==============
    if per_frame_percentiles:
        arr = np.array(per_frame_percentiles)
        global_min = float(arr[:, 0].min())
        global_max = float(arr[:, 1].max())
        print(
            f"global depth bounds across {len(per_frame_percentiles)} frames: "
            f"min={global_min:.3f}, max={global_max:.3f}"
        )
    else:
        global_min, global_max = None, None
        print("warning: no foreground found in any frame, skipping render pass")

    # ============== Pass 2: render with global bounds (no inference) ==============
    #for image_name in tqdm(image_names, desc="pass 2: render"):
    processed_depth_list = []
    for image,mask,depth in tqdm(zip(images,mask_list,depth_list), total=len(images),desc="pass 2: render"):    
        #base_path = os.path.join(args.output, image_name.rsplit(".")[0])
        #depth_npy_path = f"{base_path}_depth.npy"

        # if not os.path.exists(depth_npy_path):
        #     continue

        #image, mask = load_image_and_mask(input_dir, seg_dir, image_name)
        # if not np.any(mask):
        #     continue

        #depth = np.load(depth_npy_path).astype(np.float32)

        processed_depth = process_depth_map_with_bounds(
            depth, mask, global_min, global_max
        )
        panels = [image, processed_depth]
        if model.with_normal:
            normal_vis = compute_surface_normals(depth, mask, global_min, global_max)
            panels.append(normal_vis)

        if model.no_black_background:
            for p in panels[1:]:
                p[mask == 0] = image[mask == 0]

        #vis_image = np.concatenate(panels, axis=1)
        #cv2.imwrite(f"{base_path}{os.path.splitext(image_name)[1]}", vis_image)
        processed_depth = cv2.cvtColor(processed_depth, cv2.COLOR_BGR2RGB) 
        processed_depth = torch.from_numpy(processed_depth.astype(np.float32) / 255.0).unsqueeze(0)
        processed_depth_list.append(processed_depth)
    return processed_depth_list


def pose_predict(model,detector,images,node_cr_path,no_save_json=True,radius=3,thickness=1,kpt_thr=0.3):
        ## add pose metainfo to model

    num_keypoints = model.cfg.num_keypoints
    if num_keypoints == 308:
        model.pose_metainfo = parse_pose_metainfo(
            dict(from_file=os.path.join(node_cr_path, "sapiens/pose/configs/_base_/keypoints308.py"))
        )
    ## add codec to model
    codec_type = model.cfg.codec.pop("type")
    assert codec_type == "UDPHeatmap", "Only support UDPHeatmap"
    model.codec = UDPHeatmap(**model.cfg.codec)

    # build detector
    # detector = init_detector(args.det_config, args.det_checkpoint, device=args.device)
    # detector.cfg = mmdet_pipeline(detector.cfg)

    # Get image list
    # if os.path.isdir(args.input):
    #     input_dir = args.input
    #     image_names = [
    #         name
    #         for name in sorted(os.listdir(input_dir))
    #         if name.endswith((".jpg", ".png", ".jpeg"))
    #     ]
    # else:
    #     with open(args.input, "r") as f:
    #         image_paths = [line.strip() for line in f if line.strip()]
    #     image_names = [os.path.basename(path) for path in image_paths]
    #     input_dir = os.path.dirname(image_paths[0])

    frames_records = []
    image_size = None
    num_keypoints_seen = None
    pose_list = []
    for i, image in tqdm(enumerate(images), total=len(images)):
    #for image_name in tqdm(image_names, total=len(image_names)):
        # image_path = os.path.join(input_dir, image_name)
        # image = cv2.imread(image_path)

        try:
            keypoints, keypoint_scores, bboxes = process_one_image(
                 image, detector, model
            )
        except Exception as e:
            print(f"[vis_pose] inference failed on {i}: {e}")
            continue

        if image_size is None:
            image_size = [int(image.shape[0]), int(image.shape[1])]
        if num_keypoints_seen is None and len(keypoints) > 0:
            num_keypoints_seen = int(np.asarray(keypoints[0]).shape[0])

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis_image_rgb = visualize_keypoints(
            image=image_rgb,
            keypoints=keypoints,
            keypoints_visible=np.ones_like(keypoint_scores) > 0,
            keypoint_scores=keypoint_scores,
            radius=radius,
            thickness=thickness,
            kpt_thr=kpt_thr,
            skeleton=model.pose_metainfo["skeleton_links"],
            kpt_color=model.pose_metainfo["keypoint_colors"],
            link_color=model.pose_metainfo["skeleton_link_colors"],
        )

        vis_image_rgb=torch.from_numpy(vis_image_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        pose_list.append(vis_image_rgb)
        if not no_save_json:
            try:
                instances = []
                for kpts, scores, bbox in zip(keypoints, keypoint_scores, bboxes):
                    instances.append({
                        "bbox": [float(v) for v in np.asarray(bbox).reshape(-1)[:4]],
                        "keypoints": np.asarray(kpts, dtype=float).tolist(),
                        "keypoint_scores": np.asarray(scores, dtype=float).reshape(-1).tolist(),
                    })
                frames_records.append({
                    "image_name": str(i),
                    "instances": instances,
                })
            except Exception as e:
                 print(f"[vis_pose] json record failed on {i}: {e}")
    return pose_list




def normal_predict(model,images, mask_list,):

    image_list = []
    if mask_list is None:
        mask_list = [None] * len(images)
    for image,mask in tqdm(zip(images,mask_list), total=len(images)):
        if mask is None:
            mask = np.ones_like(image[:, :, 0], dtype=bool)
        ##------------------------------------------
        data = model.pipeline(dict(img=image))  ## resize and pad
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs, data_samples = data["inputs"], data["data_samples"]

        with torch.no_grad():
            normal = model(inputs)  # normal is 1 x 3 x H x W
            normal = normal / torch.norm(normal, dim=1, keepdim=True).clamp(
                min=1e-8
            )  # normalize to unit length

        # ------------------------------------------
        pad_left, pad_right, pad_top, pad_bottom = data_samples["meta"]["padding_size"]
        normal = normal[
            :,
            :,
            pad_top : inputs.shape[2] - pad_bottom,
            pad_left : inputs.shape[3] - pad_right,
        ]

        normal = F.interpolate(
            normal,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        normal = normal.squeeze(0).cpu().numpy().transpose(1, 2, 0)  ## H x W x 3

        normal[mask == 0] = -1
        normal_vis = ((normal + 1) / 2 * 255).astype(np.uint8)
        normal_vis = normal_vis[:, :, ::-1]

        if model.no_black_background:
            normal_vis[mask == 0] = image[mask == 0]

        normal_vis_rgb = cv2.cvtColor(normal_vis, cv2.COLOR_BGR2RGB) 
        image = torch.from_numpy(normal_vis_rgb.astype(np.float32) / 255.0).unsqueeze(0)
        image_list.append(image)
    return  image_list




def load_model(checkpoint,node_dir,device):
    model_name,model_type=Path(checkpoint).stem.rsplit('_', 1)
    print(f"Loading model: {model_name}, {model_type} from checkpoint")
    config = get_config_path(model_name,model_type, node_dir)
    model = init_model(config, checkpoint, device=device)
    return model

def get_config_path(model_name,model_type,node_dir):
    dataset_dict={
        "seg": "shutterstock_goliath",
        "pointmap": "render_people",
        "albedo": "render_people",
        "normal": "metasim_render_people",
        "pose": "metasim_render_people"
        }
    
    if model_type!="pose":
        data_dir=dataset_dict[model_type]
        model_name=f"{model_name}_{model_type}_{data_dir}-1024x768"
        config_file=os.path.join(node_dir, f"sapiens/dense/configs/{model_type}/{data_dir}/{model_name}.py")
    else:
        model_name=f"{model_name}_keypoints308_shutterstock_goliath_3po-1024x768"
        config_file=os.path.join(node_dir, f"sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/{model_name}.py")
    return config_file


def init_model(
    config: Union[str, Path],
    checkpoint: Optional[Union[str, Path]] = None,
    device: str = "cuda:0",
):
    #assert isinstance(config, (str, Path))
    assert checkpoint is None or isinstance(checkpoint, (str, Path))

    config = Config.fromfile(config)

    ## avoid loading the pretrained backbone weights
    if "init_cfg" in config.model["backbone"]:
        config.model["backbone"].pop("init_cfg")

    model = MODELS.build(config.model)
    data_preprocessor = MODELS.build(config.data_preprocessor)

    if checkpoint is not None:
        if str(checkpoint).endswith(".safetensors"):
            state_dict = load_file(checkpoint, device="cpu")
        else:  # Handle .pth and .bin files
            checkpoint_data = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            state_dict = (
                checkpoint_data["state_dict"]
                if "state_dict" in checkpoint_data
                else checkpoint_data["model"]
            )

        incompat = model.load_state_dict(state_dict, strict=False)

        if incompat.missing_keys:
            print(f"Missing keys: {incompat.missing_keys}")

        if incompat.unexpected_keys:
            print(f"Unexpected keys: {incompat.unexpected_keys}")

        print(f"\033[96mModel loaded from {checkpoint}\033[0m")

    model.cfg = config
    model.data_preprocessor = data_preprocessor
    model.pipeline = Compose(config.test_pipeline)

    model.to(device)
    model.eval()

    return model

def process_one_image(image, detector, model):
    image_w, image_h = image.shape[1], image.shape[0]
    #det_result = inference_detector(detector, image)
   

    # pred_instance = det_result.pred_instances.cpu().numpy()
    # bboxes = np.concatenate(
    #     (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    # )
    # bboxes = bboxes[
    #     np.logical_and(
    #         pred_instance.labels == 0,  ## 0 is the person class
    #         pred_instance.scores > args.bbox_thr,
    #     )
    # ]

    # bboxes = bboxes[nms(bboxes, args.nms_thr), :4]  ## B x 4; x1, y1, x2, y2
    bboxes=detector(image) # use yolo to detect the person
    # get bbox from the image size
    if bboxes is None or len(bboxes) == 0:
        bboxes = np.array([[0, 0, image_w - 1, image_h - 1]], dtype=np.float32)

    inputs_list = []
    data_samples_list = []
    for bbox in bboxes:
        data_info = dict(img=image)
        data_info["bbox"] = bbox[None]  # shape (1, 4)
        data_info["bbox_score"] = np.ones(1, dtype=np.float32)  # shape (1,)
        data = model.pipeline(data_info)
        data = model.data_preprocessor(data)
        inputs_list.append(data["inputs"])
        data_samples_list.append(data["data_samples"])

    inputs = torch.cat(inputs_list, dim=0)  # B x 3 x H x W
    with torch.no_grad():
        pred = model(inputs)  # B x 3 x H x W
        if model.cfg.val_cfg is not None and model.cfg.val_cfg.get("flip_test", False):
            pred_flipped = model(inputs.flip(-1))  # B x 3 x H x W
            pred_flipped = pred_flipped.flip(-1)  ## B x K x heatmap_H x heatmap_W
            flip_indices = model.pose_metainfo["flip_indices"]
            assert len(flip_indices) == pred_flipped.shape[1]  ## K
            pred_flipped = pred_flipped[:, flip_indices]
            pred = (pred + pred_flipped) / 2.0

    # ------------------------------------------
    pred = pred.cpu().numpy()  ## B x K x heatmap_H x heatmap_W
    keypoints = []
    keypoint_scores = []
    for i, data_samples in enumerate(data_samples_list):
        ## kps in crop image
        ## keypoints_i is 1 x K x 2
        # keypoint_scores_i is 1 x K
        keypoints_i, keypoint_scores_i = model.codec.decode(pred[i])
        input_size = data_samples["meta"]["input_size"]  ## 1 x 2, 768 x 1024
        bbox_center = data_samples["meta"]["bbox_center"]  ## 1 x 2
        bbox_scale = data_samples["meta"]["bbox_scale"]  ## 1 x 2

        keypoints_i = (
            keypoints_i / input_size * bbox_scale + bbox_center - 0.5 * bbox_scale
        )
        keypoints.append(keypoints_i[0])  ## remove fake batch dim
        keypoint_scores.append(keypoint_scores_i[0])  ## remove fake batch dim

    return keypoints, keypoint_scores, bboxes
