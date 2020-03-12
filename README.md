# Real Time Sketch Recognition

## Summary

This repository is implementation of my project for Artificial Intelligence course. In this project, AlexNet deep CNN model [1] has been utilized to classify sketch objects and TU Berlin sketch dataset [2] has been used in order to train classifier model. Moreover, shortcomings will be improved in time.

**Note:** In the next version, the classification and user interface will work in different threads. Besides, the training codes (both TensorFlow 1.x and 2.0 versions) will be uploaded soon.

## Prerequisites:
- TensorFlow 1.7 or later
- Python 3
- Tkinter 8.6.8
- Pillow 5.4.1
- OpenCV 3.1
- Pynput 1.4.2
- Numpy 1.14.2

This code is tested with Titan X GPUs.

## Pretrained model

The pretrained AlexNet model can be downloaded here:
- https://www.dropbox.com/s/4gxxv1mpu0sfrfw/sketch_model.zip?dl=0

## Dataset
- http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

## Demo
```
 python drawing_tool.py 
```

## Cite

For detail about sketch classification experiment, check [our paper.](https://ieeexplore.ieee.org/abstract/document/8404417)

For citation:

<cite>
@inproceedings{eyiokur2018sketch,
  title={Sketch classification with deep learning models},
  author={Eyiokur, Fevziye {\.I}rem and Yaman, Do{\u{g}}ucan and Ekenel, Haz{\i}m Kemal},
  booktitle={2018 26th Signal Processing and Communications Applications Conference (SIU)},
  pages={1--4},
  year={2018},
  organization={IEEE}
}
<\cite>


## Acknowledgement

The AlexNet.py script is based on [this implementation.](https://github.com/kratzert/finetune_alexnet_with_tensorflow/tree/5d751d62eb4d7149f4e3fd465febf8f07d4cea9d)

## References

[1] A. Krizhevsky, I. Sutskever, G. E. Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, Advances in Neural Information Processing Systems, 2012.

[2] M. Eitz, J. Hays, M. Alexa, *How do humans sketch objects?*, ACM Trans. Graph. 31.4:44-1, 2012.
