from src.my_ir import IRGraph
class Compiler:
    def __init__(self):
        pass 

    def compile(self, fn, example_inputs, device = None) -> IRGraph:
        graph = IRGraph(fn,example_inputs)
        graph.fusion_pass()
        return graph


    