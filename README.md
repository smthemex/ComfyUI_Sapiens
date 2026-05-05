# ComfyUI_Sapiens
You can call Using Sapiens or Sapiens2 to get seg,normal,pose,depth,mask maps,albedo maps,pointmap.

Sapiens From: [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) 
Sapiens2 From: [facebookresearch/sapiens2](https://github.com/facebookresearch/sapiens2)  

Update
----
* clean up codes ,sapiens2 support body spilit now,清理代码，2代已支持身体部分分离,使用插件自带的split节点； 


previous update
----
*  感谢@lyxkilo 的代码，通过他的代码可以将fp32的模型转为fp16模型（更小，1B seg 2G左右），首次运行会生成一个同名加fp16模型文件，生成后不需要再开启fp16生成按钮；


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
|     ├── sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2
|     ├── sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2
|     ├── sapiens_0.3b_normal_render_people_epoch_66_torchscript.pt2
|     ├── sapiens_1b_render_people_epoch_88_torchscript.pt2
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
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v2_n.png)
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v2_p.png)
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v2_s.png)
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v2_ps.png)

* sapiens1       
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v1.png)
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v1_p.png)
![](https://github.com/smthemex/ComfyUI_Sapiens/blob/main/example_workflows/v1_s.png)

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
