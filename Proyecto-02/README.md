# Compilador de Tensores Minimalista (Chiqui-IR)

Este proyecto implementa un compilador simple que traduce un lenguaje de dominio específico (DSL) para operaciones tensoriales a código Python optimizado con NumPy.

* **`chiqui_ir.py`**: Define la Representación Intermedia (Grafo Computacional, Tensores, Operaciones).
* **`codegen.py`**: Backend que traduce el IR a código fuente de Python (NumPy).
* **`main.py`**: Frontend (Parser) y orquestador. Lee el archivo fuente, construye el IR y ejecuta el código generado.
* **`programa.txt`**: Archivo de entrada con el código fuente a compilar.

Requisitos:

* Python 3.8+
* NumPy


Para compilar:
python main.py


Operaciones soportadas:
    para definir entradas:
    input <NOMBRE> <DIMENSIONES>
    Ejemplo: input A 128,64

    para operaciones:
    <VAR> = matmul <A> <B>   # Multiplicación de matrices
    <VAR> = add <A> <B>      # Suma elemento a elemento
    <VAR> = relu <A>         # Activación ReLUs