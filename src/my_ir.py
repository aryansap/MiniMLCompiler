import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
import operator


class Tensor: 
    def __init__(self, name: str, shape: list[int], dtype: str):
        self.name = name
        self.shape = shape
        self.dtype = dtype
    def __repr__(self):
        return f"Name: {self.name}, Shape: {self.shape}, Dtype: {self.dtype}"


class Op:
    def __init__(self, op_type, inputs: list[Tensor], output: Tensor, attrs: dict):
        self.op_type = op_type
        self.inputs = inputs
        self.output = output
        self.attrs = attrs
    
    def __repr__(self):
        return f"Op: {self.op_type}, Inputs: {self.inputs}, Outputs: {self.output}, attrs: {self.attrs}"

class IRGraph:
    def __init__(self, fn, example_inputs):
        #do fx tracing and lowering here, this will contain the lsit of operations
        self.ops = []
        self.tensors: dict[str, Tensor] = {}
        self.inputs: list[Tensor] = []
        self.outputs: list[Tensor] = []

        # 1) Trace
        self.gm: fx.GraphModule = fx.symbolic_trace(fn)
        self.fx_graph: fx.Graph = self.gm.graph

        # 2) Annotate nodes with shape/dtype metadata using example inputs
        #    This fills node.meta["tensor_meta"] for tensor-producing nodes.
        ShapeProp(self.gm).propagate(*example_inputs)

        # Helper: make a Tensor from an FX node using its tensor_meta
        def tensor_from_node(n: fx.Node) -> Tensor:
            tm = n.meta.get("tensor_meta", None)
            if tm is None:
                raise ValueError(
                    f"Node {n.format_node()} has no tensor_meta. "
                    f"Make sure example_inputs are torch.Tensors and ShapeProp ran successfully."
                )
            shape = list(tm.shape)
            dtype = str(tm.dtype)
            return Tensor(name=n.name, shape=shape, dtype=dtype)

        # env maps FX Node -> IR Tensor
        env: dict[fx.Node, Tensor] = {}

        # 3) Lower FX nodes to your IR
        for node in self.fx_graph.nodes:
            if node.op == "placeholder":
                t = tensor_from_node(node)
                env[node] = t
                self.inputs.append(t)
                self.tensors[t.name] = t

            elif node.op == "call_function":
                # Only support a small set of ops for now
                tgt = node.target

                if tgt is operator.matmul:
                    op_type = "MATMUL"
                elif tgt is operator.add:
                    op_type = "ADD"
                elif tgt is torch.nn.functional.gelu:
                    op_type = "GELU"
                elif tgt is torch.nn.functional.relu:
                    op_type = "RELU"
                else:
                    raise NotImplementedError(f"Unsupported call_function target: {tgt}")

                # Map FX args (Nodes) -> IR Tensors
                ir_inputs: list[Tensor] = []
                for a in node.args:
                    if isinstance(a, fx.Node):
                        ir_inputs.append(env[a])
                    else:
                        # For v1, reject constants/other Python values
                        raise NotImplementedError(f"Non-tensor arg {a} in node {node.format_node()} is not supported yet")
                out_t = tensor_from_node(node)
                ir_op = Op(op_type=op_type, inputs=ir_inputs, output=out_t, attrs={})

                self.ops.append(ir_op)
                env[node] = out_t
                self.tensors[out_t.name] = out_t

            elif node.op == "output":
                # FX output node has node.args = (return_value,)
                (rv,) = node.args

                if isinstance(rv, fx.Node):
                    self.outputs = [env[rv]]
                elif isinstance(rv, (tuple, list)):
                    self.outputs = [env[x] for x in rv]
                else:
                    raise NotImplementedError(f"Unsupported output type: {type(rv)}")
            else:
                raise NotImplementedError(f"Unsupported FX node.op: {node.op}")

    def fusion_pass(self):
        #do fusion pass on member variables here 
        #maybe do some kind of stack thing wher you have a stack of operations, 
        # once you see matmul, add, gelu, pop the 3 and add the fused verison 
        pass

    def execute(self, runtime_inputs):
        #create a map of the names of Tensors to the data
        #execute ops in the list
        pass
    def dump(self):
        print("=== IRGraph Dump ===")

        print("\nInputs:")
        for t in self.inputs:
            print(f"  {t}")

        print("\nOps:")
        for i, op in enumerate(self.ops):
            in_names = [t.name for t in op.inputs]
            out_name = op.output.name
            print(f"  [{i}] {op.op_type}({', '.join(in_names)}) -> {out_name}  attrs={op.attrs}")

        print("\nOutputs:")
        for t in self.outputs:
            print(f"  {t}")

        print("\nAll Tensors:")
        for name, t in self.tensors.items():
            print(f"  {name}: {t}")

        print("====================")

    