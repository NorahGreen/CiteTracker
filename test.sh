source /userhome/anaconda3/bin/activate citetrack
cd /code/CiteTracker
# python tracking/train.py --script citetrack --config vitb_256_mae_32x4_ep300 --save_dir ./out1 --mode single --use_wandb 0
python tracking/test.py citetrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
cd ..
