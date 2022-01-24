# Image Convolution and Manipulation
This project is built mainly on the python libraries "Tkinter", "numpy", "sci-py", and "open-cv" and serves the purpose of helping me better understand the image manipulation techniques used in convolutional neural networks by understanding how convolutions are applied and the effects of different kernel shapes and values on an image onto which it is convolved.

# External Dependencies
To run the program, install the following packages using pip, and then run the python script found in this repo.
```
Dependencies:
pip install Pillow
pip install numpy
pip install scipy
pip install opencv-python
```

# Sobel edge detection demo
custom filters/kernels can be created or default, built-in filters like the gaussian blur or the sobel filters can be used.

Steps for sobel edge detection:
1. apply the sobel Y filter to find the vertical edges.
2. apply the sobel X filter to find the horizontal edges.
3. flatten the two to get an image where brightness represents the "sharpness" of an edge.

[![sobel demo](https://user-images.githubusercontent.com/47716543/103279510-f68b1280-499b-11eb-96c9-ff25585a5065.png)](https://user-images.githubusercontent.com/47716543/103278314-33a1d580-4999-11eb-8214-d5eafd0e09ee.mp4 "Applying the sobel edge detection filters")

# Moving RGB layers
color layers can be moved around.

[![moving layers](https://user-images.githubusercontent.com/47716543/103279640-38b45400-499c-11eb-9578-3601945c6ddb.png)](https://user-images.githubusercontent.com/47716543/103278158-d1e16b80-4998-11eb-86b7-a09bf3b5e6fe.mp4 "RGB layers can be rearranged")

# Flattening image layers
Image layers can be flattened (assigning flattened layers the average value of all layers selected to be flattened. This gives a greyscale image when all three layers are flattened.

[![flattening image](https://user-images.githubusercontent.com/47716543/103279724-76b17800-499c-11eb-9885-48c391f5104f.png)](https://user-images.githubusercontent.com/47716543/103278069-9c3c8280-4998-11eb-81ec-08240e97e8e3.mp4 "image layers can be flattened, giving a greyscale image when all layers are flattened")

# Restoring the original image
The original image can be restored at any point in time, undoing all previous edits made to it.

[![restoring image](https://user-images.githubusercontent.com/47716543/103279791-a06a9f00-499c-11eb-815b-5f3fece797e0.png)](https://user-images.githubusercontent.com/47716543/103278208-f2112a80-4998-11eb-808e-312c7404bfc2.mp4 "The image can be restored to its original appearance at any time")

