python train_y_v2.py \
  --video_list "train_list_with_warping.csv" \
  --lr_width 1920 \
  --lr_height 1080 \
  --hr_width 3840 \
  --hr_height 2160 \
  --samples_per_epoch 1000 \
  --batch_size 8\
  --patch_size 192 \
  --epochs 50 \
  --val_stride 384 \
  --bit_depth 10 \
  --train_ratio 0.8\
  --resume /mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9/YUV_SR/checkpoints_y/best.pth\
  # --png_dir /mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9/train_orig_part0/train/train_orig/000/