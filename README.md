#### Resources
- Implementations of **SRDRM** and **SRDRM-GAN** for underwater image super-resolution
- Simplified implementation of SRGAN, ESRGAN, EDSRGAN, ResNetSR, SRCNN, and DSRCNN
- Implementation: TensorFlow >= 1.11.0, Keras >= 2.2, and Python 2.7
  
![img1](/data/fig1b.jpg)

#### Pointers
- Paper: https://arxiv.org/pdf/1909.09437.pdf
- USR-248 dataset: http://irvlab.cs.umn.edu/resources/usr-248-dataset
- Bibliography entry for citation:
	```
	@inproceedings{islam2018dynamic,
	  title={{Underwater Image Super-Resolution using Deep Residual Multipliers}},
	  author={Islam, Md Jahidul and Enan, Sadman Sakib and Luo, Peigen and Sattar, Junaed},
	  booktitle={To appear at the IEEE International Conference on Robotics and Automation (ICRA)},
	  year={2020},
	  organization={IEEE}
	}
	```
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

#### Acknowledgements
- https://github.com/Mulns/SuperSR
- https://github.com/david-gpu/srez
- https://github.com/wandb/superres
- https://github.com/tensorlayer/srgan
- https://github.com/icpm/super-resolution
- https://github.com/alexjc/neural-enhance
- https://github.com/jiny2001/dcscn-super-resolution
- https://github.com/titu1994/Image-Super-Resolution
- https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras




