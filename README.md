Neural Style Transfer

This project implements a Neural Style Transfer (NST) model to apply artistic styles to photographs. The model uses the VGG19 network pretrained on the ImageNet dataset to extract content and style features. By optimizing a target image, the model blends the content of one image with the style of another.

Features

Combines content and artistic style from two images.

Real-time training with CUDA support.

Adjustable style intensity and content weight.

Supports popular image formats (JPEG, PNG).

Technologies Used

Python 3.9

PyTorch

TorchVision

Pillow

Matplotlib

Acknowledgements

Inspired by the work of Gatys et al. on Neural Style Transfer.



Utilizes the VGG19 model from TorchVision.
