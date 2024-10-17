# ComfyUI_Sapiens
You can call Using Sapiens to get seg，normal，pose，depth，mask.

Sapiens From: [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) 

1.Installation  
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Sapiens
```
2.requirements  
----
```
pip install -r requirements.txt
```
If some module is missing, please pip install   #ultralytics yolov8

3 Need  model 
----
**3.1 base :(choice repo_id or ckpt_name)**       
* 3.1.1 #sapiens    
only support torchscript version now,you can choice 1b,0.3b,0.6b,2b,do not changge ckpt's name!!!    
只支持torchscript的版本，但是有多种模型可选，最好质量的是1b或者2B，如果模型选择全是none，会自动下载一个1B的seg，下载后不要改模型名字；     
[seg](https://huggingface.co/facebook/sapiens-seg-1b-torchscript)  
[pose](https://huggingface.co/facebook/sapiens-pose-1b-torchscript)  
[depth](https://huggingface.co/facebook/sapiens-depth-1b-torchscript)  
[normal](https://huggingface.co/facebook/sapiens-normal-1b-torchscript)  
```
├── ComfyUI/models/sapiens/
|     ├── seg/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2
|     ├── pose/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2
|     ├── normal/sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2
|     ├── depth/sapiens_1b_render_people_epoch_88_torchscript.pt2
```
* 3.1.2 yolo # if using pose
yolo是pose必须的，官方的太复杂，不如yolo好用；  
[yolov8m](https://huggingface.co/Ultralytics/YOLOv8/tree/main)   
```
├── ComfyUI/models/sapiens/
|     ├── yolov8m.pt
```
4 Example
----
**seg body**    
* 最新示例，可以选人体部位28种类型 （Latest version)        
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example.png)   

Citation
------
* Using some ibaiGorordo's code from [ibaiGorordo](https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference)  
* UsingJaykumaran's codes from [Jaykumaran](https://learnopencv.com/sapiens-human-vision-models/）
**facebookresearch/sapiens**
``` python  
@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```
