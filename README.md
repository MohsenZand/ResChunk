# [Multiscale Residual Learning of Graph Convolutional Sequence Chunks for Human Motion Prediction](https://arxiv.org/abs/2308.16801)
## Abstract:
A new method is proposed for human motion prediction by learning temporal and spatial dependencies.
Recently, multiscale graphs have been  developed to model the human body at higher  abstraction levels, resulting in more stable motion prediction. 
Current methods however predetermine scale levels and combine spatially proximal joints to generate coarser scales based on human priors, even though movement patterns in different motion sequences vary and do not fully comply with a fixed graph of spatially connected joints. Another problem with graph convolutional methods is mode collapse, in which predicted poses converge around a mean pose with no discernible movements, particularly in long-term predictions. To tackle these issues, we propose \emph{ResChunk}, an end-to-end network which explores dynamically correlated body components based on the pairwise relationships between all joints in individual sequences. ResChunk  is trained to learn the residuals between target sequence chunks in an autoregressive manner to enforce the temporal connectivities between consecutive chunks. 
It is hence a sequence-to-sequence prediction network which considers dynamic spatio-temporal features of sequences at multiple levels. 
Our experiments on two challenging benchmark datasets, CMU Mocap and Human3.6M, demonstrate that our proposed method is able to effectively model the sequence information for motion prediction and outperform other techniques to set a new state-of-the-art.

## Datasets
CMU Mocap can be downloaded from [here](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics/tree/master/data/cmu_mocap).

H3.6M can be downloaded from [here](https://github.com/una-dinosauria/human-motion-prediction).

Once downloaded, datasets must be processed using 'preprocess_datasets.py'


## Training 
All models are implemented in 'models.py'.
To train our method on each dataset, 'db' flag in 'flag_sets.py' must be set as the name of dataset (cmu or h36m). 
After setting the 'PATH' for 'data_dir' and 'run_dir', the model can be trained by simply running 'train.py'.
Other flags in 'flag_sets.py' can be changed if needed. 


## Evaluation and Visualization
To evaluate and visualize the results, the corresponding flags in 'EVALUATION' section of 'flag_sets.py' must be set. Then, running 'evaluation.py' can generate the results and save the figures and videos in the 'run_dir'. 

## Citation
Please cite our papers if you use code from this repository:
```
@article{zand2023multiscale,
  title={Multiscale Residual Learning of Graph Convolutional Sequence Chunks for Human Motion Prediction},
  author={Zand, Mohsen and Etemad, Ali and Greenspan, Michael},
  journal={arXiv preprint arXiv:2308.16801},
  year={2023}
}
```

```
@article{zand2023flow,
  title={Flow-Based Spatio-Temporal Structured Prediction of Motion Dynamics},
  author={Zand, Mohsen and Etemad, Ali and Greenspan, Michael},
  booktitle={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={11},
  pages={1--13},
  year={2023},
  publisher={IEEE}
}
```

