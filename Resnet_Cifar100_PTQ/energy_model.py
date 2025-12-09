import torch
import torch.nn as nn

# -----------------------------------------------------------
# MAC estimation functions
# -----------------------------------------------------------

def conv2d_macs(module, inp, out):
    Cin = module.in_channels
    Cout = module.out_channels
    Kh, Kw = module.kernel_size
    Hout, Wout = out.shape[2], out.shape[3]
    groups = module.groups
    return Cout * Hout * Wout * (Cin // groups) * Kh * Kw


def linear_macs(module, inp, out):
    return module.in_features * module.out_features


MAC_FNS = {
    nn.Conv2d: conv2d_macs,
    nn.Linear: linear_macs,
}

# -----------------------------------------------------------
# Memory access estimate
# -----------------------------------------------------------
def estimate_mem_access(module):
    params = 0
    for p in module.parameters():
        params += p.numel()
    return params


# -----------------------------------------------------------
# Energy models
# -----------------------------------------------------------
def E_mac(bitwidth):
    return bitwidth ** 2   # quadratic cost


def E_mem(bitwidth):
    return bitwidth         # linear cost


# -----------------------------------------------------------
# Compute energy using forward hooks
# -----------------------------------------------------------
@torch.no_grad()
def compute_energy(model, bitwidth_map, input_size=(1,3,32,32), device="cpu"):
    model.eval()
    model = model.to(device)

    macs_per_layer = {}
    mem_per_layer = {}

    # Hook registration
    def register_hooks():
        for name, module in model.named_modules():
            if type(module) in MAC_FNS:
                def hook(mod, inp, out, name=name):
                    macs = MAC_FNS[type(mod)](mod, inp[0], out)
                    macs_per_layer[name] = macs
                    mem_per_layer[name] = estimate_mem_access(mod)
                module.register_forward_hook(hook)

    register_hooks()

    dummy = torch.randn(*input_size).to(device)
    _ = model(dummy)

    total_energy = 0
    layer_energy = {}

    for layer_name in macs_per_layer:
        b = bitwidth_map[layer_name]
        mac_e = macs_per_layer[layer_name] * E_mac(b)
        mem_e = mem_per_layer[layer_name] * E_mem(b)
        layer_energy[layer_name] = mac_e + mem_e
        total_energy += layer_energy[layer_name]

    return total_energy, layer_energy
