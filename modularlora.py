class ModularLoRALinear(Module):
"""Linear ModularLoRA Layer"""

def __init__(self, ...):
    self.hatWq_z_s = quantize(pretrained_W)
    (self.A, self.B) = lora_init(...)
    
def forward(self, x):
    (hatWq, z, s) = self.hatWq_z_s
    return LPLinear.apply(x, hatWq, z, s) \ + (x @ self.B)
@ self.A.t() + self.bias
