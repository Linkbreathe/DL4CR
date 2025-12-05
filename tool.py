import torch

# Example: loading a pretraining checkpoint (contains optimizer, args, etc.)
# ckpt_path = "./tg3k/checkpoints_mim/mim_best.pth"
# This checkpoint has keys:
# dict_keys(['epoch', 'model_state_dict', 'optimizer_state_dict', 'args'])

# Load a finetuned UNet checkpoint (raw state_dict only)
ckpt_path = "./tg3k/checkpoints_finetune/mim/best_model.pth"

# Load checkpoint to CPU
ckpt = torch.load(ckpt_path, map_location="cpu")

# For a finetuned UNet, ckpt is usually a pure state_dict:
# odict_keys([
#   'inc.double_conv.0.weight', 'inc.double_conv.1.weight', 'inc.double_conv.1.bias',
#   'inc.double_conv.1.running_mean', 'inc.double_conv.1.running_var',
#   'inc.double_conv.1.num_batches_tracked',
#   ...
#   'down1.maxpool_conv.1.double_conv.0.weight',
#   ...
#   'up4.conv.double_conv.4.running_var',
#   'up4.conv.double_conv.4.num_batches_tracked',
#   'outc.conv.weight', 'outc.conv.bias'
# ])
# These keys directly correspond to the UNet2D layer names.

print(type(ckpt))     # Should print <class 'collections.OrderedDict'> for a raw state_dict
print(ckpt.keys())    # Prints all top-level parameter names in the checkpoint
