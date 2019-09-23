### Resources
- Implementations of **SRDRM** and **SRDRM-GAN** for underwater image super-resolution
- Simplified implementation of SRGAN, ESRGAN, EDSRGAN, ResNetSR, SRCNN, and DSRCNN
- Implementation: TensorFlow >= 1.11.0, Keras >= 2.2, and Python 2.7
  
| Single Image Super-Resolution (SISR) | Color and sharpness   | 
|:--------------------|:--------------------|
| ![det-1a](/data/fig1b.jpg) | ![det-1b](/data/col.jpg) | 

### Pointers
- Paper: https://arxiv.org/pdf/1909.09437.pdf
- USR-248 dataset: http://irvlab.cs.umn.edu/resources/usr-248-dataset
- Bibliography entry for citation:
	```
	article{islam2019srdrm,
	    title={Underwater Image Super-Resolution using Deep Residual Multipliers},
	    author={Islam, Md Jahidul and Enan, Sadman Sakib and Luo, Peigen and Sattar, Junaed},
	    journal={arXiv preprint arXiv:1909.09437},
	    year={2019}
	}
	```
- Video demo: https://youtu.be/qOLZVgrxCwE

| 2x SISR performance | 4x SISR performance  | 
|:--------------------|:--------------------|
| ![det-enh](/data/2x.gif) | ![det-gif](/data/4x.gif) |

#### Usage
- Download the data, setup data-paths in the training scripts
- Use the individual scripts for training 2x, 4x, 8x SISR models 
	- train-GAN-nx.py: SRDRM-GAN, SRGAN, ESRGAN, EDSRGAN
	- train-generative-models-nx.py: SRDRM, ResNetSR, SRCNN, DSRCNN
	- Checkpoints: checkpoints/dataset-name/model-name/
	- Samples: images/dataset-name/model-name/
- Use the test-scripts for evaluating different models
	- A few test images: data/test/ (ground-truth: high_res)
	- Output: data/output/ 
- A few saved models are provided in checkpoints/saved/
- Use the [measure.py](measure.py) for quantitative analysis based on UIQM, SSIM, and PSNR 

#### Constraints and Challenges
- Trade-offs between performance and running time. Requirements:
	- Running time >= 5 FPS on Jetson-TX2 
	- Model size <= 12MB (no quantization) 
- Challenges
	- Performance for 8x models
	- Inconsistent coloring, infrequent noisy patches

### Acknowledgements
- https://github.com/wandb/superres
- https://github.com/david-gpu/srez
- https://github.com/Mulns/SuperSR
- https://github.com/tensorlayer/srgan
- https://github.com/icpm/super-resolution
- https://github.com/alexjc/neural-enhance
- https://github.com/jiny2001/dcscn-super-resolution
- https://github.com/titu1994/Image-Super-Resolution
- https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras




