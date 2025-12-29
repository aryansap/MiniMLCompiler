import torch
import torch.nn.functional as F

from src.compiler import Compiler
from src.my_ir import IRGraph

# ---- Simple test function ----
def simple_fn(x, W, b):
    return F.gelu(x @ W + b)


def main():
    # Example inputs (compile-time shapes)
    x = torch.randn(4, 8)
    W = torch.randn(8, 16)
    b = torch.randn(16)

    example_inputs = (x, W, b)

    # Build IR
    ir = IRGraph(simple_fn, example_inputs)

    # Dump IR
    ir.dump()


if __name__ == "__main__":
    main()
