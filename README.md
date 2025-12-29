# Mini ML Compiler

A toy ML compiler built on top of Torch FX that lowers PyTorch programs into a
custom intermediate representation (IR) with shape-aware ops.

## Features
- Torch FX tracing
- Shape propagation
- Custom IR (Tensor + Op)
- Supported ops:
  - MatMul
  - Add (with broadcasting)
  - GELU
  - ReLU
- Planned:
  - Pattern-based op fusion (MatMul → Add → GELU)
  - Custom fused CUDA kernels
  - Simple runtime executor
