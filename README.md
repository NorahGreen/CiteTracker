## CiteTracker
The official implementation for the **iccv 2023** paper [_CiteTracker: Correlating Image and Text for Visual Tracking_](https://arxiv.org/abs/2308.11322).

[[Models](https://drive.google.com/drive/folders/12byllgwhJQVBS6EK7XnfQU7m_9JuDCCZ?usp=sharing)][[Raw Results]([https://drive.google.com/drive/folders/1TYU5flzZA1ap2SLdzlGRQDbObwMxCiaR?usp=sharing](https://drive.google.com/drive/folders/12byllgwhJQVBS6EK7XnfQU7m_9JuDCCZ?usp=sharing)][[Data](https://drive.google.com/drive/folders/1TtHNzc4ils5yjAi5bIXZA3nK3iRNpszB?usp=drive_link)]

<p align="center">
  <img width="85%" src="https://github.com/NorahGreen/CiteTracker/blob/main/fig/framework.png" alt="Framework"/>
</p>

## Install the environment
**Option1**: Use the Anaconda (CUDA 10.2)
```
conda create -n citetrack python=3.8
conda activate citetrack
bash install.sh
```
**Option2**: Use the Anaconda (CUDA 11.3)
```
conda env create -f environment.yaml
```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like this:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

## Training
Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script citetrack --config vitb_384_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 4 --use_wandb 1
```

Replace `--config` with the desired model config under `experiments/citetrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights from [Models](https://drive.google.com/drive/folders/12byllgwhJQVBS6EK7XnfQU7m_9JuDCCZ?usp=sharing)

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/citetrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py citetrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py citetrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name citetrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```
- TrackingNet
```
python tracking/test.py citetrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name citetrack --cfg_name vitb_384_mae_ce_32x4_ep300
```

## Acknowledgments
* Thanks [OSTrack](https://github.com/botaoye/OSTrack) and [COCOOP](https://github.com/KaiyangZhou/CoOp/tree/main) libraries for helping us to quickly implement our ideas.
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.  

## Citation
```
@inproceedings{citetracker,
  title={CiteTracker: Correlating Image and Text for Visual Tracking},
  author={Li, Xin and Huang, Yuqing and He, Zhenyu and Wang, Yaowei and Lu, Huchuan and Yang, Ming-Hsuan},
  booktitle={ICCV},
  year={2023}
}
