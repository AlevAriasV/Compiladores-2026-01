# chiqui_ir.py
from typing import List, Optional

class Node:
    """Clase base para cualquier nodo en el grafo."""
    def __init__(self, name: str):
        self.name = name

class Tensor(Node):
    """Representa un dato."""
    def __init__(self, name: str, shape: List[int], dtype: str = "float32"):
        super().__init__(name)
        self.shape = shape
        self.dtype = dtype

    def __repr__(self):
        return f"Tensor({self.name}, {self.shape})"

class Operation(Node):
    """Clase base para operaciones."""
    def __init__(self, name: str, inputs: List[Node]):
        super().__init__(name)
        self.inputs = inputs

class AddOp(Operation):
    """Suma"""
    pass

class MatMulOp(Operation):
    """Multiplicación de matrices"""
    pass

class ReluOp(Operation):
    """Función de activación ReLU: max(0, x)"""
    pass

class IRBuilder:
    """Ayudante para construir el grafo y manejar nombres únicos."""
    def __init__(self):
        self.ops: List[Operation] = []
        self.counter = 0

    def new_var_name(self, prefix="t"):
        name = f"{prefix}{self.counter}"
        self.counter += 1
        return name

    def input(self, name, shape):
        return Tensor(name, shape)

    def add(self, a: Node, b: Node) -> Node:
        # Crea la operación, asigna un nombre al resultado y guárdala
        out_name = self.new_var_name("add_res")
        op = AddOp(out_name, [a, b])
        self.ops.append(op)
        return op # La operación actúa como el tensor de salida

    def matmul(self, a: Node, b: Node) -> Node:
        out_name = self.new_var_name("mm_res")
        op = MatMulOp(out_name, [a, b])
        self.ops.append(op)
        return op

    def relu(self, a: Node) -> Node:
        out_name = self.new_var_name("relu_res")
        op = ReluOp(out_name, [a])
        self.ops.append(op)
        return op