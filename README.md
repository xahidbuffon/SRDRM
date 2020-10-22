*Repository for the paper [Underwater Image Super-Resolution using Deep Residual Multipliers](https://ieeexplore.ieee.org/document/9197213) (ICRA 2020).  [Pre-print](https://arxiv.org/pdf/1909.09437.pdf).*
![img1](/data/fig1b.jpg)

#### Resources
- Proposed dataset: [USR-248](http://irvlab.cs.umn.edu/resources/usr-248-dataset)
- Proposed model: **SRDRM** and **SRDRM-GAN** for underwater image super-resolution 
- Models in comparison: SRGAN, ESRGAN, EDSRGAN, ResNetSR, SRCNN, and DSRCNN
- Requirements: TensorFlow >= 1.11 and Keras >= 2.2


#### Usage
- Download the data, setup data-paths in the training scripts
	- train-GAN-nx.py: SRDRM-GAN, SRGAN, ESRGAN, EDSRGAN
	- train-generative-models-nx.py: SRDRM, ResNetSR, SRCNN, DSRCNN
- Use the test-scripts for evaluating different models
	- A few test images: data/test/ (ground-truth: high_res)
- Use the [measure.py](measure.py) for quantitative analysis


#### Bibliography Entry

	@inproceedings{islam2020srdrm,
	  title={{Underwater Image Super-Resolution using Deep Residual Multipliers}},
	  author={Islam, Md Jahidul and Enan, Sadman Sakib and Luo, Peigen and Sattar, Junaed},
	  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
	  year={2020},
	  organization={IEEE}
	}
	

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


