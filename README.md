# 2HPs_demonstrator

## how to pipeline
1. generate 1hp-boxes dataset on pcsgs05 (now in shared directory)
2. connect to ipvsgpu1, then move 1hp-boxes dataset to scratch/sgs/pelzerja (`cd ~/pelzerja/test_nn/dataset_generation_laptop/Phd_simulation_groundtruth`, `)
3. prepare 1hp-boxes dataset on ipvsgpu1
4. run training on ipvsvgpu01 with this dataset
5. copy trained model and prepared dataset to lapsgs29
6. generate large-domain dataset on pcsgs05
7. copy large-domain dataset to lapsgs29
8. run pipeline on lapsgs29 with this dataset and trained model and the respective 1hp-boxes dataset that the model was trained with


## NEW


## how to pipeline
1. generate 1hp-boxes dataset on pcsgs05 (now in shared directory)
2. connect to ipvsgpu1, then move 1hp-boxes dataset to scratch dir (`cd ~/pelzerja/test_nn/dataset_generation_laptop/Phd_simulation_groundtruth`, `mv NAME_DATASET/ /scratch/sgs/pelzerja/datasets/1hp_boxes/NAME_DATASET/`)

3. generate large-domain dataset on pcsgs05
4. copy large-domain dataset to lapsgs29 into `datasets/2hps_demonstrator`
5. connect to ipvsgpu1, then move large-domain dataset from pcsgs05 to ipvsgpu/scratch (`cd ~/pelzerja/test_nn/dataset_generation_laptop/Phd_simulation_groundtruth`, `mv NAME_DATASET/ /scratch/sgs/pelzerja/datasets/2hps_demonstrator_copy_of_local/NAME_DATASET/`)

3. prepare 1hp-boxes dataset on ipvsgpu1
4. run training on ipvsgpu1 with this dataset
5. copy trained model and prepared dataset to lapsgs29
8. run pipeline on lapsgs29 with this dataset and trained model and the respective 1hp-boxes dataset that the model was trained with


## which datasets:
currently available and correct (5x5m cells, inverse SDf, assumed steady state, ...)

### iso perm:
1hp-boxes: ipvsgpu1/scratch/../1hp_boxes/benchmark_dataset_2d_100datapoints_grad_p
model: current_unet_benchmark_dataset_2d_100datapoints_grad_p
large-domain: lapsgs29/../datasets/2hps_demonstrator/benchmark_large_2hps_iso_perm (_close/far/very_far)

### vary perm:
1hp-boxes: ipvsgpu1/scratch/../1hp_boxes/benchmark_dataset_2d_1000dp_vary_perm
model: current_unet_benchmark_dataset_2d_1000dp_vary_perm
large-domain: lapsgs29/../datasets/2hps_demonstrator/benchmark_large_2hps_vary_perm (_close/far)

## others
1hp-boxes: ipvsgpu1/scratch/../1hp_boxes/benchmark_dataset_2d_100dp_vary_perm