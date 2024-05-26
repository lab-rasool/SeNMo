# SeNMo
A Self-Normalizing DL Model for Enhanced Multi-Omics Data Analysis in Oncology

## Running the SeNMo Training

To run the SeNMo Training with the specified parameters, use the following command:

```bash
python SeNMo_Training.py \
    --regression True \
    --finetune False  \
    --exp_name surv \
    --reg_type all \
    --disease pancancer_combined \
    --task surv \
    --gpu_ids 0 \
    --lr 0.0005811726189177087 \
    --weight_decay 0.005978947728252338 \
    --dropout_rate 0.10583716299176746 \
    --batch_size 256 \
    --dataroot <> \
    --checkpoints_dir <> \



