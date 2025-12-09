import torch
import torch.nn as nn
import torch.nn.functional as F

class UniformQuantizer(nn.Module):
    """
    Symmetric uniform quantizer for weights and activations.
    """
    def __init__(self, bitwidth=8, per_channel=False):
        super().__init__()
        self.bitwidth = bitwidth
        self.per_channel = per_channel

    def forward(self, x):
        if self.bitwidth == 32:
            return x

        qmin = -(2 ** (self.bitwidth - 1))
        qmax = (2 ** (self.bitwidth - 1)) - 1

        # Per-channel weight quantization (Conv2d only)
        if self.per_channel and x.dim() == 4:
            max_val = x.abs().amax(dim=(1,2,3), keepdim=True)
        else:
            max_val = x.abs().max()

        scale = max_val / qmax
        scale = torch.clamp(scale, min=1e-8)

        x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
        return x_q * scale

# HOOKS FOR PTQ
def weight_quant_hook(module, inputs):
    """
    This hook fires BEFORE the forward of Conv2d/Linear.
    It replaces FP32 weights (and bias) with quantized versions.
    """
    if hasattr(module, "weight_quant"):
        module.weight.data = module.weight_quant(module.weight.data)

    if module.bias is not None and hasattr(module, "bias_quant"):
        module.bias.data = module.bias_quant(module.bias.data)


def activation_quant_hook(module, inputs, output):
    """
    This hook fires AFTER forward and quantizes activation tensors.
    """
    if hasattr(module, "act_quant"):
        return module.act_quant(output)
    return output

# Attach quantizers to model
def quantize_model(model, bitwidth):
    """
    Add weight and activation quantization to the model using hooks.
    """
    # Attach weight quantization
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):

            # Weight + bias quantizers
            module.weight_quant = UniformQuantizer(bitwidth, per_channel=True)
            if module.bias is not None:
                module.bias_quant = UniformQuantizer(bitwidth)

            # MAIN FIX: Quantize weights during forward pass
            module.register_forward_pre_hook(weight_quant_hook)

    # Attach activation quantizers (ReLU family)
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
            module.act_quant = UniformQuantizer(bitwidth, per_channel=False)
            module.register_forward_hook(activation_quant_hook)

    return model
