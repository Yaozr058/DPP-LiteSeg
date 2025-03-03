# DPP-LiteSeg: A real-time semantic segmentation model of urban street scenes
DPP LiteSeg is a semantic segmentation model for urban landscapes, which is based on the Pp LiteSeg framework to better analyze detailed and contextual information. Combining Adaptive Efficient Channel Attention (AECA) with the use of DeSTDCNet backbone network. In addition, we used the YOLO+Mosaic data augmentation method for small-scale datasets. The DPP LiteSeg series achieves an excellent balance between inference speed and accuracy. Specifically, on the small dataset CamVid, DPP LiteSeg with DeSTDC1 backbone achieved an average joint crossover (mIoU) of 78.21% at 162.3 frames per second (FPS), while using DeSTDC2 backbone achieved 80.45% mIoU at 156.1 FPS.
## All content will be uploaded after the official accepted of the paper（DPP-LiteSeg.py DeSTDC1.pth DeSTDC2.pth）
## platform
My platform is like this:

* Win11
* nvidia 4070 gpu
* cuda 12.6
* cudnn 9.6
* python 3.12.3
* TensorRT 10.7
* pytorch 2.5.0


## Prepare Dataset
* camvid
Download the dataset from the official [website](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). Then decompress them into the `data/camvid` directory:  
```
    |-data
       |-CamVid
         |-test
         |-test_labels
         |-train
         |-train_labels
         |-val
         |-val_labels
```
OR [website](https://github.com/Yaozr058/DPP-LiteSeg/tree/data)
## Train
Training commands I used to train the models (python .\Train_DPPLiteSeg.py) 
```
 !!!we provided is not complete and only displays partial content. Please adjust according to your training needs!!!
```
## Demo
We provide a demo to quickly use the model (python .\Demo.py)
## Networking Framework
Taking DPP-LiteSeg-L as an example(DeSTDC1 as the backbone network)
[D2STDC1](https://github.com/user-attachments/assets/cf5d7d0d-80c0-425b-85bf-c3fdb1e4146d)
## Inference Results
``` Input Image ```
![image](https://github.com/user-attachments/assets/d1005a9d-69dc-4701-8f24-f88af04348e3)
``` DeSTDC1 ```
![image](https://github.com/user-attachments/assets/1e5f9e30-2fbc-4242-bdce-eeffa7f41262)
``` DeSTDC2 ```
![image](https://github.com/user-attachments/assets/601b469e-e697-4c2f-b3da-a0df32b81ae3)
* The inference results may be affected by the inference environment and equipment
