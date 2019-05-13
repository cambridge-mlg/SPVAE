# SPVAE
Research code for ICML 2019 paper on "Hierarchical Decompositional Mixtures of Variational Autoencoders"


## Requirements
* python3
* tensorflow > 1.8


## Main Idea of Paper
Variational autoencoders (VAEs) have received considerable attention, since they allow us to learn expressive neural density estimators effectively and efficiently. However, learning and inference in VAEs is still problematic due to the sensitive interplay between the generative model and the inference network. Since these problems become generally more severe in high dimensions, we propose a novel hierarchical mixture model over low-dimensional VAE experts. Our model decomposes the overall learning problem into many smaller problems, which are coordinated by the hierarchical mixture, represented by a sum-product network. In experiments we show that our model consistently outperforms classical VAEs on all of our experimental benchmarks. Moreover, we show that our model is highly data efficient and degrades very gracefully in extremely low data regimes.


## Download from Github
`.npy` files are stored using Git LFS. 
Before cloning this directory, please install Git LFS according to the instructions
on [this link](https://github.com/git-lfs/git-lfs/wiki/Installation).
If you are using an ubuntu system like me, a convenience script `install_gitlfs_ubuntu.sh` is provided.

Without Git LFS installed, `.npy` files will still be downloaded 
but will not be interpretable by numpy.  

If Git LFS cannot be installed, the dataset can still be obtained by running one of the
`download_and_process_*.py` scripts. 


## Project Layout

* `src` is where the low level logic of SPN and VAEs are implemented.
* `data` houses the three different datasets used in the paper in a consistent format
* `scripts` contains blueprints for how different components are joined together
* `main` is where models are run and analysed

 
## Getting Started
Run the `hptune*.py` hyperparameter tuning scripts that are within the `main` directory.
The results will be saved within the same directory and can be compiled using the `compile*.ipynb` notebooks.

Use of GPUs is highly recommended. Otherwise the models will take too long to train.
Expected duration of hyperparameter tuning on a single GPU: 1 week


