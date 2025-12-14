# codegen.py
from chiqui_ir import AddOp, MatMulOp, ReluOp, Operation, Tensor

class PythonGenerator:
    def __init__(self):
        self.buffer = []
        self.indent = "    "

    def generate(self, ops: list[Operation], inputs: list[Tensor], func_name="modelo_generado"):
        # 1. Cabecera e imports
        self.buffer.append("import numpy as np")
        self.buffer.append("")
        
        # 2. Definición de la función
        input_names = ", ".join([inp.name for inp in inputs])
        self.buffer.append(f"def {func_name}({input_names}):")
        
        # 3. Cuerpo de la función (instrucciones)
        for op in ops:
            line = self._dispatch(op)
            self.buffer.append(f"{self.indent}{line}")
        
        # 4. Return (asumimos que la última op es el resultado)
        if ops:
            last_var = ops[-1].name
            self.buffer.append(f"{self.indent}return {last_var}")
        
        return "\n".join(self.buffer)

    def _dispatch(self, op: Operation) -> str:
        """Convierte una Op de IR a string de Python."""
        if isinstance(op, AddOp):
            # t0 = A + B
            lhs = op.inputs[0].name
            rhs = op.inputs[1].name
            return f"{op.name} = {lhs} + {rhs}"
            
        elif isinstance(op, MatMulOp):
            # t0 = np.matmul(A, B)
            lhs = op.inputs[0].name
            rhs = op.inputs[1].name
            return f"{op.name} = np.matmul({lhs}, {rhs})"
            
        elif isinstance(op, ReluOp):
            # t0 = np.maximum(0, A)
            inp = op.inputs[0].name
            return f"{op.name} = np.maximum(0, {inp})"
            
        else:
            raise NotImplementedError(f"Operación {type(op)} no soportada")