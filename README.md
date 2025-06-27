# Diferenciação Automática com Grafos Computacionais (Mini-PyTorch)

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

Este projeto é uma implementação de um motor de **Diferenciação Automática (Autograd)** do zero, utilizando Python e NumPy. O objetivo é replicar a funcionalidade central de frameworks modernos de Deep Learning como o PyTorch, construindo um sistema capaz de criar grafos computacionais dinâmicos e calcular gradientes através de backpropagation (reverse-mode automatic differentiation).

Este trabalho foi desenvolvido como parte dos requisitos da disciplina de Inteligência Artificial, com base na especificação do "Trabalho 1: Diferenciação Automática com Grafos Computacionais".

## Conceitos Fundamentais

O framework é construído sobre dois pilares do Deep Learning:

1.  **Grafos Computacionais:** Toda sequência de operações matemáticas é representada como um grafo acíclico dirigido (DAG). As variáveis (Tensores) são os nós, e as operações são as arestas que conectam esses nós. Isso nos permite rastrear todas as dependências entre as variáveis.

2.  **Backpropagation (Diferenciação Automática Reversa):** Após calcular o resultado final (a "perda") em uma passada para frente (forward pass), o algoritmo de backpropagation percorre o grafo no sentido inverso. Ele aplica a **Regra da Cadeia** da calculus em cada nó para calcular eficientemente o gradiente da saída final em relação a cada um dos parâmetros de entrada.

## Arquitetura do Projeto

A estrutura do código foi projetada para ser modular e extensível:

-   **`Tensor`**: A classe central que encapsula um array `numpy`. Ela armazena o dado, se requer gradiente (`requires_grad`), seu gradiente acumulado (`grad`), e as operações que a geraram (`_parents` e `operation`).
-   **`Op`**: Uma classe base abstrata que define a interface para qualquer operação. Toda operação deve implementar um método `__call__` (para o forward pass) e um método `grad` (para o backward pass).
-   **Operações Concretas**: Classes como `Add`, `MatMul`, `ReLU`, etc., que herdam de `Op` e implementam a lógica matemática específica para suas passadas forward e backward.
-   **`NameManager`**: Uma classe utilitária para dar nomes únicos e legíveis aos tensores gerados, facilitando a depuração.

## Funcionalidades Implementadas

-   **Grafo Computacional Dinâmico:** O grafo é construído em tempo real à medida que as operações são executadas.
-   **Motor de Backpropagation:** O método `Tensor.backward()` inicia a propagação de gradientes a partir de qualquer nó do grafo.
-   **Acúmulo de Gradientes:** Os gradientes são somados, e não substituídos, a cada chamada de `.backward()`, permitindo o uso correto em arquiteturas complexas e no treinamento com mini-batches.
-   **Biblioteca de Operações Essenciais:** Suporte para um vasto conjunto de operações necessárias para construir redes neurais.

## Operações Suportadas

O framework suporta as seguintes operações matemáticas e de ativação:

| Álgebra Linear | Trigonométricas | Ativações | Outras |
| :--- | :--- | :--- | :--- |
| `Add` | `Sin` | `ReLU` | `Sum` (`my_sum`) |
| `Sub` | `Cos` | `Sigmoid` | `Mean` |
| `Prod` (Element-wise) | `Tanh` | `Softmax` | `Square` |
| `MatMul` (Matrix Mult) | | | `Exp` |


### Exemplo de Uso

A utilização do framework é intuitiva. Tensores são criados e então combinados através das funções de operação. Para calcular os gradientes, basta chamar o método `.backward()` no tensor final.

```python
# Exemplo 1: Derivada de uma multiplicação
a = Tensor([1.0, 2.0, 3.0])
b = Tensor([4.0, 5.0, 6.0])

# c = a * b
c = prod(a, b) 

# d = c * 3.0
d = prod(c, 3.0) 

# Inicia o backpropagation a partir de 'd'
d.backward()

# Imprime o gradiente de 'd' em relação a 'a' (d(d)/da)
# Esperado: 3.0 * b = [12, 15, 18]
print("Gradiente em a:", a.grad) 

# Imprime o gradiente de 'd' em relação a 'b' (d(d)/db)
# Esperado: 3.0 * a = [3, 6, 9]
print("Gradiente em b:", b.grad)
```

## Executando os Testes

O notebook fornecido contém uma seção de "Testes Básicos" que valida a implementação da derivada para cada operação individualmente. Você pode executar cada célula naquela seção para verificar a corretude do framework.
