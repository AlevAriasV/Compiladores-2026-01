# main.py
import numpy as np
import importlib.util
import sys
import os

from chiqui_ir import IRBuilder
from codegen import PythonGenerator

def parse_and_compile(filename):
    builder = IRBuilder()
    variables = {} # Diccionario para guardar referencias: "nombre" -> Objeto Nodo

    print(f"--- Leyendo archivo fuente: {filename} ---")
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        command = parts[0]

        # 1. Manejo de inputs: input NOMBRE DIM1,DIM2
        if command == "input":
            name = parts[1]
            dims = [int(x) for x in parts[2].split(',')]
            # Creamos el tensor en el IR y lo guardamos en el diccionario
            variables[name] = builder.input(name, dims)
            print(f"Parsed Input: {name} shape={dims}")

        # 2. Manejo de asignaciones: res = op arg1 arg2
        elif len(parts) >= 3 and parts[1] == "=":
            dest_name = parts[0]  # Nombre de la variable destino
            op_type = parts[2]    # matmul, add, relu
            
            # Buscamos los operandos en nuestro diccionario de variables
            arg1_name = parts[3]
            arg1_node = variables[arg1_name]
            
            if op_type == "matmul":
                arg2_name = parts[4]
                arg2_node = variables[arg2_name]
                # Llamamos al builder
                op_node = builder.matmul(arg1_node, arg2_node)
            
            elif op_type == "add":
                arg2_name = parts[4]
                arg2_node = variables[arg2_name]
                op_node = builder.add(arg1_node, arg2_node)
                
            elif op_type == "relu":
                op_node = builder.relu(arg1_node)
            
            else:
                raise ValueError(f"Operación desconocida: {op_type}")
            
            # Sobrescribimos el nombre interno del nodo para que coincida
            # con el nombre de variable que el usuario puso en el archivo de texto
            op_node.name = dest_name 
            variables[dest_name] = op_node
            print(f"Parsed Op: {dest_name} = {op_type}(...)")

    return builder, variables

def main():
    # 1. FRONTEND: Parsing
    # leemos el archivo externo
    source_file = "programa.txt"
    if not os.path.exists(source_file):
        print(f"Error: Crea el archivo '{source_file}' primero.")
        return

    builder, vars_dict = parse_and_compile(source_file)
    
    # Identificamos cuáles son inputs para el generador
    inputs_list = [v for k, v in vars_dict.items() if hasattr(v, 'shape') and not hasattr(v, 'inputs')]

    print("\n=== 2. BACKEND: Generando Código Python ===")
    gen = PythonGenerator()
    codigo_python = gen.generate(builder.ops, inputs=inputs_list)
    print(codigo_python)

    # 3. EJECUCIÓN DINÁMICA
    module_name = "kernel_generado"
    file_path = f"{module_name}.py"
    with open(file_path, "w") as f:
        f.write(codigo_python)

    # Carga dinámica
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        print("\n=== 3. PRUEBA: Ejecutando con datos aleatorios ===")
        # Generamos datos aleatorios basados en los inputs leídos
        args = []
        print("Entradas generadas:")
        for inp in inputs_list:
            data = np.random.rand(*inp.shape).astype(np.float32)
            args.append(data)
            print(f" - {inp.name}: {inp.shape}")

        try:
            res = module.modelo_generado(*args)
            print(f"\nResultado final shape: {res.shape}")
            print("¡Compilación y ejecución exitosa!")
        except Exception as e:
            print(f"Error en ejecución: {e}")

if __name__ == "__main__":
    main()