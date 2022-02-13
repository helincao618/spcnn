# Semantic-Panoptic-Completion

The thesis implemented a self-supervised approach that convert partial and noisy RGB-D scans into high-quality 3D scene reconstructions with semantic labels by a sparse convolutional autoencoder.

### Install
1. This implementation uses Python 3.6, [Pytorch1.7.1](http://pytorch.org/), cudatoolkit 11.0. We recommend to use [conda](https://docs.conda.io/en/latest/miniconda.html) to deploy the environment.
   * Install with conda:
    ```
    conda env create -f environment.yml
    conda activate rfdnet
    ```
    * Install with pip:
    ```
    pip install -r requirements.txt
    ```


## Installation:  
This thesis uses Python 3.8, Pytorch 1.4.0, cudatoolkit 10.0. We recommend to use conda to deploy the environment.

    ```
    conda create -n spc python=3.8
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
    pip install plyfile h5py scipy
    ```

The thesis also use [SparseConvNet](https://github.com/facebookresearch/SparseConvNet). Please install it in your virtual environment.

    ```
    export CUDA_HOME=/usr/local/cuda-10.0
    
    git clone https://github.com/facebookresearch/SparseConvNet.git
    
    cd SparseConvNet/
    
    bash develop.sh
    ```

For visualization, please install Marching Cubes

    ```
    cd torch/marching_cubes
    
    python setup.py install
    ```

## Data:
You can download the data, or generate using the script in `datageneration`.
### Download
* Scene data: 
  - [mp_sdf_vox_2cm_input.zip](http://kaldir.vc.in.tum.de/adai/SGNN/mp_sdf_vox_2cm_input.zip) (44G)
  - [mp_sdf_vox_2cm_target.zip](http://kaldir.vc.in.tum.de/adai/SGNN/mp_sdf_vox_2cm_target.zip) (58G)
* Train data:
  - [completion_blocks.zip](http://kaldir.vc.in.tum.de/adai/SGNN/completion_blocks.zip) (88G)
### Generation
* [GenerateScans](datagen/GenerateScans) depends on the [mLib](https://github.com/niessner/mLib) library.
* [GenerateSemantic](datagen/GenerateSemantic)
## Training:  
* See `python train.py --help` for all train options. 
* Example command: 

    ```
    python train.py --gpu 0 --data_path ./data/completion_blocks --semantic_data_path ./data/h5_semantic_train_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --start_epoch 0 --save_epoch 1 --save logs/mp --max_epoch 4
    ```

* Trained model: [spc.pth](http://kaldir.vc.in.tum.de/adai/SGNN/sgnn.pth) (7.5M)

### Testing
* See `python test_scene.py --help` for all test options. 
* Example command: 

    ```
    python test_scene.py --gpu 0 --input_data_path ./data/mp_sdf_vox_2cm_input --target_data_path ./data/mp_sdf_vox_2cm_target --test_file_list ../filelists/mp-rooms_val-scenes.txt --model_path sgnn.pth --output ./output  --max_to_vis 20
    ```

## Reference
[https://github.com/angeladai/sgnn](https://github.com/angeladai/sgnn)
[https://github.com/facebookresearch/SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
