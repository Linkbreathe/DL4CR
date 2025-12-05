# finetune_all.ps1
# finetune：MIM(unet/scag/nac) + FullImage(unet/scag/nac)

$python = "D:\Tool\Anaconda3\envs\dlcv\python.exe"

# $python = "python"

# ---------------- MIM: unet ----------------
Write-Host "===== [1/6] MIM finetune - unet ====="
& $python ".\tg3k_finetune_segementation.py" --images-dir ".\tg3k\tg3k\thyroid-image" --masks-dir ".\tg3k\tg3k\thyroid-mask" --split-json ".\tg3k\tg3k\tg3k-trainval.json" --init-from ".\tg3k\checkpoints_unet_mim\mim_best.pth" --tensorboard-dir ".\tg3k\runs\mim_unet_finetune_tg3k" --save-dir ".\tg3k\checkpoints_finetune\mim_unet" --train-subset-size 500 --model-type "unet"

# ---------------- MIM: scag ----------------
Write-Host "===== [2/6] MIM finetune - scag ====="
& $python ".\tg3k_finetune_segementation.py" --images-dir ".\tg3k\tg3k\thyroid-image" --masks-dir ".\tg3k\tg3k\thyroid-mask" --split-json ".\tg3k\tg3k\tg3k-trainval.json" --init-from ".\tg3k\checkpoints_scag_mim\mim_best.pth" --tensorboard-dir ".\tg3k\runs\mim_scag_finetune_tg3k" --save-dir ".\tg3k\checkpoints_finetune\mim_scag" --train-subset-size 500 --model-type "scag"

# ---------------- MIM: nac ----------------
Write-Host "===== [3/6] MIM finetune - nac ====="
& $python ".\tg3k_finetune_segementation.py" --images-dir ".\tg3k\tg3k\thyroid-image" --masks-dir ".\tg3k\tg3k\thyroid-mask" --split-json ".\tg3k\tg3k\tg3k-trainval.json" --init-from ".\tg3k\checkpoints_nac_mim\mim_best.pth" --tensorboard-dir ".\tg3k\runs\mim_nac_finetune_tg3k" --save-dir ".\tg3k\checkpoints_finetune\mim_nac" --train-subset-size 500 --model-type "nac"

# ---------------- FullImage: unet ----------------
Write-Host "===== [4/6] FullImage finetune - unet ====="
& $python ".\tg3k_finetune_segementation.py" --images-dir ".\tg3k\tg3k\thyroid-image" --masks-dir ".\tg3k\tg3k\thyroid-mask" --split-json ".\tg3k\tg3k\tg3k-trainval.json" --init-from ".\tg3k\checkpoints_unet_fullimage\fullimage_best.pth" --tensorboard-dir ".\tg3k\runs\fullimage_unet_finetune_tg3k" --save-dir ".\tg3k\checkpoints_finetune\fullimage_unet" --train-subset-size 500 --model-type "unet"

# ---------------- FullImage: scag ----------------
Write-Host "===== [5/6] FullImage finetune - scag ====="
& $python ".\tg3k_finetune_segementation.py" --images-dir ".\tg3k\tg3k\thyroid-image" --masks-dir ".\tg3k\tg3k\thyroid-mask" --split-json ".\tg3k\tg3k\tg3k-trainval.json" --init-from ".\tg3k\checkpoints_scag_fullimage\fullimage_best.pth" --tensorboard-dir ".\tg3k\runs\fullimage_scag_finetune_tg3k" --save-dir ".\tg3k\checkpoints_finetune\fullimage_scag" --train-subset-size 500 --model-type "scag"

# ---------------- FullImage: nac ----------------
Write-Host "===== [6/6] FullImage finetune - nac ====="
& $python ".\tg3k_finetune_segementation.py" --images-dir ".\tg3k\tg3k\thyroid-image" --masks-dir ".\tg3k\tg3k\thyroid-mask" --split-json ".\tg3k\tg3k\tg3k-trainval.json" --init-from ".\tg3k\checkpoints_nac_fullimage\fullimage_best.pth" --tensorboard-dir ".\tg3k\runs\fullimage_nac_finetune_tg3k" --save-dir ".\tg3k\checkpoints_finetune\fullimage_nac" --train-subset-size 500 --model-type "nac"

Write-Host "===== 全部 6 个 finetune 任务结束（不代表都成功，只是都跑完了） ====="
