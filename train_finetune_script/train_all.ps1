# train_all_simple.ps1
# run six scripts in row：MIM(unet/scag/nac) + full_image(unet/scag/nac)

$ErrorActionPreference = "Stop"

# using python.exe current env if using miniconda etc.
$python = "D:\Tool\Anaconda3\envs\dlcv\python.exe"


$splitJson = ".\tg3k\tg3k\tg3k-trainval.json"

# ---------- MIM Pretrain ----------
Write-Host "===== 1) MIM Pretrain - unet ====="
& $python ".\tg3k_MIM_Pretrain.py" `
  --images-dir ".\tg3k\tg3k\thyroid-image" `
  --checkpoint-dir ".\tg3k\checkpoints_unet_mim" `
  --log-dir ".\tg3k\runs\mim_tg3k_unet" `
  --model-type "unet" `
  --split-json $splitJson

Write-Host "===== 2) MIM Pretrain - scag ====="
& $python ".\tg3k_MIM_Pretrain.py" `
  --images-dir ".\tg3k\tg3k\thyroid-image" `
  --checkpoint-dir ".\tg3k\checkpoints_scag_mim" `
  --log-dir ".\tg3k\runs\mim_tg3k_scag" `
  --model-type "scag" `
  --split-json $splitJson

Write-Host "===== 3) MIM Pretrain - nac ====="
& $python ".\tg3k_MIM_Pretrain.py" `
  --images-dir ".\tg3k\tg3k\thyroid-image" `
  --checkpoint-dir ".\tg3k\checkpoints_nac_mim" `
  --log-dir ".\tg3k\runs\mim_tg3k_nac" `
  --model-type "nac" `
  --split-json $splitJson

# ---------- Full-image Pretrain ----------
Write-Host "===== 4) Full-image Pretrain - unet ====="
& $python ".\tg3k_full_image_Pretrain.py" `
  --images-dir ".\tg3k\tg3k\thyroid-image" `
  --checkpoint-dir ".\tg3k\checkpoints_unet_fullimage" `
  --log-dir ".\tg3k\runs\fullimage_tg3k_unet" `
  --model-type "unet" `
  --split-json $splitJson

Write-Host "===== 5) Full-image Pretrain - scag ====="
& $python ".\tg3k_full_image_Pretrain.py" `
  --images-dir ".\tg3k\tg3k\thyroid-image" `
  --checkpoint-dir ".\tg3k\checkpoints_scag_fullimage" `
  --log-dir ".\tg3k\runs\fullimage_tg3k_scag" `
  --model-type "scag" `
  --split-json $splitJson

Write-Host "===== 6) Full-image Pretrain - nac ====="
& $python ".\tg3k_full_image_Pretrain.py" `
  --images-dir ".\tg3k\tg3k\thyroid-image" `
  --checkpoint-dir ".\tg3k\checkpoints_nac_fullimage" `
  --log-dir ".\tg3k\runs\fullimage_tg3k_nac" `
  --model-type "nac" `
  --split-json $splitJson

Write-Host "===== All six tasks finished ====="
