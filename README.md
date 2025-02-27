# DPP-LiteSeg: A real-time semantic segmentation model of urban street scenes

## platform
My platform is like this:

* Win11
* nvidia 4070 gpu
* cuda 12.6
* cudnn 9.6
* python 3.12.3
* TensorRT 10.7
* pytorch 2.5.0


## prepare dataset
* camvid
Download the dataset from the official [website](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/). Then decompress them into the `data/camvid` directory:  
data
   --CamVid
      --test
      --test_labels
      --train
      --train_labels
      --val
      --val_labels  
