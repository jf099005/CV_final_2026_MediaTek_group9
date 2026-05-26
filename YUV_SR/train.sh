root="/mnt/20F408ADF408876E/114_2/computer_vision/CV_final_2026_MediaTek_group9/"
python train_y_sr.py \
  --lr_yuv "${root}/bitstream/base/odd_H2_H3_AMS05_27_0_5.layer0.yuv" \
  --hr_yuv "${root}/orgYUV/odd_H2_H3_AMS05_3840x2160_10bit_420_HLG.yuv" \
  --lr_width 1920 \
  --lr_height 1080 \
  --hr_width 3840 \
  --hr_height 2160 \
  --num_frames 30 \
  --val_lr_yuv "${root}/bitstream/base/odd_H2_WalkInPark_27_0_4.layer0.yuv" \
  --val_hr_yuv "${root}/orgYUV/odd_H2_WalkInPark_3840x2160_10_60fps_HLG.yuv" \
  --val_num_frames 30 \
  --batch_size 32 \
  --patch_size 96 \
  --epochs 50 \
  # --hr_png_dir "${root}/train_orig_part0/train/train_orig/000"