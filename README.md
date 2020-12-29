# Image-manipulation
This project is built mainly on the python libraries "Tkinter", "numpy", "sci-py", and "open-cv" and serves the purpose of helping me better understand the image manipulation techniques used in convolutional neural networks by understanding how convolutions are applied and the effects of different kernel shapes and values on an image onto which it is convolved.

# Sobel edge detector demo
custom filters/kernels can be created or default, built-in filters like the gaussian blur of the sobel filters can be used.

Steps for sobel edge detection:
1. apply the sobel Y filter to find the vertical edges.
2. apply the sobel x filter to find the horizontal edges.
3. flatten the two to get an image where brightness represents the "sharpness" of an edge.

[![sobel demo](https://user-images.githubusercontent.com/47716543/103278307-2f75b800-4999-11eb-989e-728ce1e12bee.png)](https://user-images.githubusercontent.com/47716543/103278314-33a1d580-4999-11eb-8214-d5eafd0e09ee.mp4 "Applying the sobel edge detection filters")

# Moving RGB layers
color layers can be moved around.

[![moving layers](https://user-images.githubusercontent.com/47716543/103278196-eaea1c80-4998-11eb-9d41-f213d922bf1d.png)](https://user-images.githubusercontent.com/47716543/103278158-d1e16b80-4998-11eb-86b7-a09bf3b5e6fe.mp4 "RGB layers can be rearranged")

# Flattening image layers
Image layers can be flattened (assigning flattened layers the average value of all layers selected to be flattened. This gives a greyscale image when all three layers are flattened.

[![flattening image](https://user-images.githubusercontent.com/47716543/103278140-c3934f80-4998-11eb-9023-cafde6f82b41.png)](https://user-images.githubusercontent.com/47716543/103278069-9c3c8280-4998-11eb-81ec-08240e97e8e3.mp4 "image layers can be flattened, giving a greyscale image when all layers are flattened")

# Restoring the original image
The original image can be restored at any point in time, undoing all previous edits made to it.

[![restoring image](https://user-images.githubusercontent.com/47716543/103278254-10772600-4999-11eb-9127-db5fa609eff6.png)](https://user-images.githubusercontent.com/47716543/103278208-f2112a80-4998-11eb-808e-312c7404bfc2.mp4 "The image can be restored to its original appearance at any time")

