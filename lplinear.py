class LPLinear(Function):
"""Low-Precision Linear Map"""

@staticmethod
def forward(ctx, input, hatWq, z, s):
  ctx.save_for_backward(hatWq, z, s)
  hatW = dequantize(hatWq, z, s) output = input @ 
hatW.t()
  return output # hatW is deallocated

@staticmethod
def backward(ctx, grad_output):
  hatWq, z, s = ctx.saved_tensors # we recompute hatW
  hatW = dequantize(hatWq, z, s)
  grad_input = grad_output @ hatW # here hatW can be deallocated
  return grad_input, None, None, None
