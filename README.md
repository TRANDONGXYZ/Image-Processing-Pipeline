# Image Processing Pipeline

In this project, I have implemented many algorithms related to image processing such as changing _brightness_, changing _contrast_, _denoising_ and _deblurring_. I have combined these methods into a complete _pipeline_ to process a single image or even multiple images. I used _OpenCV_ to implement most of the image processing algorithms, so the processing speed and results are very good.

## 1. Image procesisng algorithms
### Changing brightness
- Adaptive Gamma Correction.

### Changing contrast
- GHE.
- CLAHE.

### Deblurring
- Variance of Laplacian.
- Sharpen Filter.
- Adaptive Sharpen Filter.
- Unsharp Masking.

### Denoising
- Mean Filter.
- Gaussian Filter.
- Median Filter.
- Unsharp Filter.
- Bilateral Filter.
- Mean Filter (GPU).
- Gaussian Filter (GPU).

## 2. Project structure
The folder structure of this project:
```text
.
|-- input_images
|   |-- test.jpg
|   |-- test1.jpg
|   `-- test2.jpg
|-- normalization
|   |-- brightness.py
|   |-- colorspace.py
|   |-- contrast.py
|   |-- deblurring.py
|   `-- denoising.py
|-- datamover.py
|-- estimator.py
|-- example.py
|-- pipeline.py
`-- utils.py
```

- `input_images` folder contains several images for testing purpose.
- `normalization` folder contains algorithms for processing images like adjusting brightness, adjusting contrast, denoising and deblurring.
- `datamover.py` file contains a class called `DataMover`. It will help us to move data from CPU to GPU and vice versa to process on different device.
- `estimator.py` file has a base class called `Estimator`. It will help us to write sub class faster and easier.
- `example.py` file contains some examples to show how to use the pipeline.
- `pipeline.py` includes implementation of `NormalizationPipeline` class.
- `utils.py` has some helpful functions like getting current time and printing out some messeage to screen.

## 3. Special thing
In this project, I've tried to implement as general as possible. I've tried to create not just a pipeline of methods, but also _pipeline of pipelines_. It will be very helpful in many situations when you want to group some methods to one group.

## 4. Some explaination
- Both `pipeline` or class of methods are called `estimator`.
- The structure when you create a pipeline: list of tuple of 2 elements inluding `step_name` and `estimator`. `step_name` means the name of the step, it indicates what do you want to do in this step. `estimator` is an object.
- If you want to process images, you just call `forward` function of pipeline and pass list of images as an argument to the function.
- You can turn on `verbose` for printing out some information like execution time, `measure_time` for measure time of running pipeline.
