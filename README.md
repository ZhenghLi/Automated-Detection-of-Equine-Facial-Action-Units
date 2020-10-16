# Automated Detection of Equine Facial Action Units 
Modified models for our paper Automated Detection of Equine Facial Action Units
Environment:
* pytorch >= 1.0

## How to Run
Modify `config.py` for configuration
Call `python main.py` for training and test for the AU binary classification in cropped eye/lower face regions.

## Modified Models Detection in Pre-defined Regions of Interest
We modified the AlexNet and DRML for the binary classification in cropped eye/lower face regions.
The repos of the original models we referred to can be found at:
AlexNet: https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
DRML: https://github.com/AlexHex7/DRML_pytorch

## ROI Detector
We employed Yolov3-tiny for ROI Detector:
https://github.com/ultralytics/yolov3
Since Yolov3-tiny does not work well for small objects, we first detected the face regions and then detected eye/lower face regions from cropped face regions.
Remember to change the filters and classes in the yolov3-tiny.cfg for the detection of one class
