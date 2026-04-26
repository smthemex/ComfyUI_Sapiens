# ComfyUI_Sapiens
You can call Using Sapiens or Sapiens2 to get seg,normal,pose,depth,mask maps,albedo maps,pointmap.

Sapiens From: [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) 
Sapiens2 From: [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)  

Update
----
* Add support for sapiens2,albedo checkpoint is not available yet, 支持sapiens2，注意albedo 目前没有

previous update
----

* 基于COCOfullbody编码 ，单独使用pose模型时，可选pose的5种分离模式，分别是躯干，下肢，手，上肢，头部，对应选择seg_select 的编号分别是（21.torso，4.Left_foot,5.Left_Hand,6.Left_lower_arm,3.Face_Neck），这5种也可以自由组合，全选默认输出所有pose；
* Based on COCOfullbody encoding, when using the pose model alone, five separation modes of pose can be selected, namely ' torso, lower limbs, hands,lower_arm, and head.' The corresponding selection numbers for 'seg_ select' are (21. Torso, 4. Left_Foot, 5. Left_Hand,6.Left_lower_arm, 3. Face_Neck). These five modes can also be freely combined, and selecting all will output all poses by default;  
* Fixed bug where SEG cannot be used to separate normals, poses, and depths, and added a button to save pose npy files;
* try add MPS support or no cuda user..
* Add model uninstallation code for easy connection to other nodes. Thanks to @lyxkilo's code, it is possible to convert the FP32 model to an FP16 model (smaller, around 1B SEG 2G)，The first run will generate an fp16 model file with the same name, and there is no need to enable the fp16 generation button after generation.
* 修复无法利用SEG分离法线，姿态和深度的bug，新增保存姿态npy文件按钮; 
*  加入模型卸载代码，便于连接其他节点，感谢@lyxkilo 的代码，通过他的代码可以将fp32的模型转为fp16模型（更小，1B seg 2G左右），首次运行会生成一个同名加fp16模型文件，生成后不需要再开启fp16生成按钮；
* seg选择人体部位的方式是数字加“，”，例如 2，1，11，注意逗号是英文符号。

1.Installation  
-----
  In the ./ComfyUI/custom_nodes directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_Sapiens
```
2.requirements  
----
```
pip install -r requirements.txt
```
If some modules missing, please pip install   #ultralytics yolov8

3.checkpoints 
----
    
* 3.1 #sapiens    
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

* 3.1 #sapiens2
[pointmap,pose,seg,normal,albedo](https://huggingface.co/collections/facebook/sapiens2)
```
├── ComfyUI/models/sapiens/
|     ├── don't change the name
```

* 3.3 yolo # if using pose must use it ,yolo是pose必须的，官方的太复杂，不如yolo好用；  
[yolov8m](https://huggingface.co/Ultralytics/YOLOv8/tree/main)   
```
├── ComfyUI/models/sapiens/
|     ├── yolov8m.pt
```


4 Example
----

**seg body**    
* sapiens2
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/examplev2.png)

* 可以选人体部位27种类型 Latest version        
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/exampleA.png)
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/exampleB.png)
* only pose 仅使用pose模型分离躯干；  
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example.png) 

Citation
------
sapiens1 model citation
* Using some ibaiGorordo's codes from [ibaiGorordo](https://github.com/ibaiGorordo/Sapiens-Pytorch-Inference)  
* UsingJaykumaran's codes from [Jaykumaran](https://learnopencv.com/sapiens-human-vision-models)
  
**facebookresearch/sapiens**
``` 
@article{khirodkar2024sapiens,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar, Rawal and Bagautdinov, Timur and Martinez, Julieta and Zhaoen, Su and James, Austin and Selednik, Peter and Anderson, Stuart and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```
**facebookresearch/sapiens2**
```
@article{khirodkarsapiens2,
  title={Sapiens2},
  author={Khirodkar, Rawal and Wen, He and Martinez, Julieta and Dong, Yuan and Su, Zhaoen and Saito, Shunsuke},
  journal={arXiv preprint arXiv:2604.21681},
  year={2026}
}
```
