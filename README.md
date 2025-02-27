# DPP-LiteSeg: A real-time semantic segmentation model of urban street scenes
platform
My platform is like this:

WIN10
nvidia 4070 gpu
cuda 12.6
cudnn 9.6
python 3.12.3
TensorRT 10.7
pytorch 2.5.0


prepare dataset
1.cityscapes
Register and download the dataset from the official website. Then decompress them into the directory:data/cityscapes

$ mv /path/to/leftImg8bit_trainvaltest.zip datasets/cityscapes
$ mv /path/to/gtFine_trainvaltest.zip datasets/cityscapes
$ cd data/cityscapes
$ unzip leftImg8bit_trainvaltest.zip
$ unzip gtFine_trainvaltest.zip
2.camvid Download the dataset from the official website. Then decompress them into the directory:data/camvid

3.Check if the paths contained in lists of data/list are correct for dataset images.

train
Training commands I used to train the models can be found in here.

eval pretrained models
You can evaluate a trained model like this:

$ python tools/eval_city.py --config configs/cityscapes.py --weight-path /path/to/your/weight.pth
or

$ python tools/eval_camvid.py --cfg configs/camvid.yaml
