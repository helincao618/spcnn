# 3D Semantic Panoptic Completion for RGB-D Scans

The thesis implemented a self-supervised approach that convert partial and noisy RGB-D scans into high-quality 3D scene reconstructions with semantic labels by a sparse convolutional autoencoder.

## Installation:  
1. This thesis uses Python 3.8, Pytorch 1.4.0, cudatoolkit 10.0. We recommend to use conda to deploy the environment.
    ```
    conda create -n spcnn python=3.8
    conda activate spcnn
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.0 -c pytorch
    pip install plyfile h5py scipy tqdm matplotlib
    ```

2. The thesis also use [SparseConvNet](https://github.com/facebookresearch/SparseConvNet). Please install it in your virtual environment.
    ```
    export CUDA_HOME=/usr/local/cuda-10.0
    git clone https://github.com/facebookresearch/SparseConvNet.git
    cd SparseConvNet/
    bash develop.sh
    ```

3. For visualization, please install Marching Cubes
    ```
    cd torch/marching_cubes
    python setup.py install
    ```

## File Structure:
We recommend using the following file structure.

```
Workspace
    ├── dataset
    ├── logs
    ├── output
    ├── spcnn
    └── SparseConvNet
```

## Data:
### Completion
You can download the data, or generate using the script in `datageneration/GenerateScans`.
#### Download
* Scene data: 
  - [mp_sdf_vox_2cm_input.zip](http://kaldir.vc.in.tum.de/adai/SGNN/mp_sdf_vox_2cm_input.zip) (44G)
  - [mp_sdf_vox_2cm_target.zip](http://kaldir.vc.in.tum.de/adai/SGNN/mp_sdf_vox_2cm_target.zip) (58G)
* Train data:
  - [completion_blocks.zip](http://kaldir.vc.in.tum.de/adai/SGNN/completion_blocks.zip) (88G)
#### Generation
* [GenerateScans](datageneration/GenerateScans) depends on the [mLib](https://github.com/niessner/mLib) library.

### Semantic Segmentation
You can generate the semantic data using the scripts in [GenerateSemantic](datageneration/GenerateSemantic) with the following steps.
1. Download the `region segmentations` from [Matterport3D](https://niessner.github.io/Matterport/)
2. Re-organize the file structure as follow
```
dataset
    ├── room_mesh
    |       ├── building 0
    |       |       ├── region0.ply
    |       |       ├── region0.fsegs
    |       |       ├── region0.vsegs
    |       |       ├── region0.semseg
    |       |       ├── ...
    |       |       └── ...
    |       ├── building 1
    |       ├── ...
    |       └── ...
    ├── ...
    └── ...
```
3. Extract the training scenes using the scripts `semantic_trainscene_extraction.py`. Example command: 

    ```
    python semantic_trainscene_extraction.py --output_dir ../../../dataset/h5_semantic_scenes_extraction/ --input_dir ../../../dataset/room_mesh/
    ```
4. Extract the training blocks using the scripts `semantic_trainblock_generation.py`. Example command: 

    ```
    python semantic_trainblock_generation.py --output_dir ../../../dataset/h5_semantic_train_blocks/ --input_scene_dir ../../../dataset/mp_sdf_vox_2cm_target/ --input_block_dir ../../../dataset/completion_blocks/ --input_semantic_dir ../../../dataset/h5_semantic_scenes_extraction/
    ```
5. Extract the test scenes using the scripts `semantic_testscene_extraction.py`. Example command: 

    ```
    python semantic_testscene_extraction.py --output_dir ../../../dataset/h5_semantic_groundtruth_scenes/ --target_dir ../../../dataset/mp_sdf_vox_2cm_target/ --mesh_dir ../../../dataset/room_mesh/
    ```
## Model:
### Training
* See `python train.py --help` for all train options. 
* Example command: 

    ```
    python train.py --gpu 0 --data_path ../../dataset/completion_blocks --semantic_data_path ../../dataset/h5_semantic_train_blocks --train_file_list ../filelists/train_list.txt --val_file_list ../filelists/val_list.txt --start_epoch 0 --save_epoch 1 --save ../../logs/mp --max_epoch 4
    ```

* Trained model: [spcnn.pth](https://drive.google.com/file/d/181jxSqdDnrfbA2328QOBS4jmh-rqdwez/view?usp=sharing) (29.5M)
* To retrain the trained model, you need to use the following argument.  

    ```
    --retrain spcnn.pth --encoder_dim 16 --coarse_feat_dim 32 --refine_feat_dim 32
    ```
### Test
* See `python test_scene.py --help` for all test options. 
* Example command: 

    ```
    python test_scene.py --gpu 0 --input_data_path ../../dataset/mp_sdf_vox_2cm_input --target_data_path ../../dataset/mp_sdf_vox_2cm_target --test_file_list ../filelists/mp-rooms_test-scenes.txt --model_path spcnn.pth --output ../../output/mp  --max_to_vis 400
    ```
* To test the trained model, you need to use the following argument.  

    ```
    --encoder_dim 16 --coarse_feat_dim 32 --refine_feat_dim 32
    ```
### Evaluation
* The completion result could be printed by the script `test_scene.py` 

* To evaluate the semantic segmentation, you need to first generate the result using the script `test_scene.py` And then use the scripts `evaluation_semantic.py` 
    
    ```
    python evaluation_semantic.py --output_dir ../../../dataset/semantic_prediction/ --groundtruth_dir ../../../dataset/h5_semantic_groundtruth_scenes/ --prediction_dir ../../../output/mp/ --mesh_dir ../../../dataset/room_mesh/
    ```
## Reference
[https://github.com/angeladai/sgnn](https://github.com/angeladai/sgnn)

[https://github.com/facebookresearch/SparseConvNet](https://github.com/facebookresearch/SparseConvNet)
