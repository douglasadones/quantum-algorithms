# useful additional packages
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
# import exact as exact  # Esse módulo está dando problema

from qiskit_aer import Aer
from qiskit.tools.visualization import plot_histogram
from qiskit.circuit.library import TwoLocal
from qiskit_optimization.applications import Maxcut, Tsp
from qiskit.algorithms.minimum_eigensolvers import SamplingVQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA
from qiskit.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer

""" Obs: Todos os prints que adicionei para melhor entendimento começam com "Adicionado Manualmente:" """


def draw_graph(G, colors, pos):  # O G já espera um networkx graph
    default_axes = plt.axes(frameon=True)  # Mostra o contorno retangular que delimita as coordenadas.
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)  # Configurações do designer e informações do grafo.
    edge_labels = nx.get_edge_attributes(G, "weight")  # Recebe um dicionário com as informações das arestas do grafo. (Não entendi a relevância do 'name: weith' na documentação, mas aparentemente ele é o nome do atributo. Pode até ser alterado mas não omitido).
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)  # Mais configurações sobre o designer do grafo.
    plt.show()  # ------------------------------------------------------------------------------ adicionei isso aqui para mostrar o gráfico.


# Generating a graph of 3 nodes
n = 3  # ordem do grafo
num_qubits = n**2  # Matriz de adjacencia do grafo (Complexidade quadrática)
tsp = Tsp.create_random_instance(n, seed=123)  # n são os nós e a seed gera uma coordenada aleatória para os nós.
adj_matrix = nx.to_numpy_array(tsp.graph)  # Cria uma array usando os dados do tsp (N² do nosso grafo valorado)
print(f'Adicionado Manualmente: {adj_matrix.ndim}')  # Adicionei esta linha apenas para ter certeza que a array é bidimensional.
print("distance\n", adj_matrix)

colors = ["r" for node in tsp.graph.nodes]  # Os nós terão a cor vermelha r -> red
pos = [tsp.graph.nodes[node]["pos"] for node in tsp.graph.nodes]  # Lista de tuplas contendo as coordenadas de cada nó
print(f'Adicionado Manualmente: {pos}')  # Adicionei esta linha só para verificar as posições.
draw_graph(tsp.graph, colors, pos)  # GUI usando a biblioteca plt.

# Brute force approach --> Tentar todas as possibilidades possíveis uma por uma.

from itertools import permutations


def brute_force_tsp(w, N):  # 'w' é a matriz adj (formato array bidimensional) e o N é a ordem.
    a = list(permutations(range(1, N)))  # retorna o número de permutações entre 1 e 2 (No nosso caso)
    print(f'Adicionado Manualmente: all permutations: {a}')
    last_best_distance = 1e10  # Isso seria o parâmetro livre 'A' da fórmula?
    for i in a:  # loop para todos as permutações geradas por 'a' na linha 45. Isso aqui que garante passar por todas as combinações possíveis.
        print(i)
        distance = 0
        pre_j = 0  # Isso garante que começaremos com o nó 0. A primeira interação abaixo é com a permutação (1, 2), logo, ele irá verificar a distância entre os nós 0 e 1 e dps 1 e 2
        for j in i:
            print(f'Adicionado Manualmente: current step: {w[j, pre_j]}. end-node label: {j}')
            distance = distance + w[j, pre_j]  # retorna a distância entre os nós j e pre_j e soma na distancia total
            print(f'Adicionado Manualmente: Distance of step: {distance}')
            pre_j = j  # passemos para o nó seguinte
        print(f'Adicionado Manualmente: Distance between last pre_j and 0: {w[pre_j, 0]}')
        print(f'Adicionado Manualmente: current distance sum: {distance}')
        distance = distance + w[pre_j, 0]  # Esta linha garante a soma do última passo feito até o nó 0 (já que precisamos voltar para a cidade inicial).
        order = (0,) + i  # armazena os passos feitos
        if distance < last_best_distance:  # Condicional para garantir o armazenamento da ordem da menor distância
            best_order = order
            last_best_distance = distance
            print("order = " + str(order) + " Distance = " + str(distance))
    return last_best_distance, best_order


best_distance, best_order = brute_force_tsp(adj_matrix, n)
print(
    "Best order from brute force = "
    + str(best_order)
    + " with total distance = "
    + str(best_distance)
)


def draw_tsp_solution(G, order, colors, pos):
    G2 = nx.DiGraph()  # directed Graph
    G2.add_nodes_from(G)  # add nodes from G
    n = len(order)  # number of nodes in graph
    for i in range(n):  # i and j from W(i,j)
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])  # add directed edges with its weights
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(  # attributes of nodes and edges in the graph
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")  # get the weight from edges
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)
    plt.show()  # ------------------------------------------------------------------------------------- adicionei isso aqui para mostrar o gráfico.


draw_tsp_solution(tsp.graph, best_order, colors, pos)


# Mapping to the Ising problem

print('Adicionado Manualmente: Here # Mapping to the Ising problem')

qp = tsp.to_quadratic_program()  # Convert into class ~qiskit_optimization.problems.QuadraticProgram
print(qp.prettyprint())  # "Returns a pretty printed string of this problem."


# from qiskit_optimization.converters import QuadraticProgramToQubo
#
# qp2qubo = QuadraticProgramToQubo()
# qubo = qp2qubo.convert(qp)
# qubitOp, offset = qubo.to_ising()
# print("Offset:", offset)
# print("Ising Hamiltonian:")
# print(str(qubitOp))
#
# # result = exact.solve(qubo)  # Tive que comentar isso para evitar o problema do módulo exact
# # print(result.prettyprint())
#
# # Checking that the full Hamiltonian gives the right cost
#
# # Making the Hamiltonian in its full form and getting the lowest eigenvalue and eigenvector
# ee = NumPyMinimumEigensolver()
# result = ee.compute_minimum_eigenvalue(qubitOp)
#
# print("energy:", result.eigenvalue.real)
# print("tsp objective:", result.eigenvalue.real + offset)
# x = tsp.sample_most_likely(result.eigenstate)
# print("feasible:", qubo.is_feasible(x))
# z = tsp.interpret(x)
# print("solution:", z)
# print("solution objective:", tsp.tsp_value(z, adj_matrix))
# draw_tsp_solution(tsp.graph, z, colors, pos)
#
#
# # Daqui em diante dá o mesmo problema de "main thread is not in main loop" envolvendo o módulo plt

# # Running it on quantum computer
#
# algorithm_globals.random_seed = 123
# seed = 10598
#
#
# optimizer = SPSA(maxiter=300)
# ry = TwoLocal(qubitOp.num_qubits, "ry", "cz", reps=5, entanglement="linear")
# vqe = SamplingVQE(sampler=Sampler(), ansatz=ry, optimizer=optimizer)
#
# result = vqe.compute_minimum_eigenvalue(qubitOp)
#
# print("energy:", result.eigenvalue.real)
# print("time:", result.optimizer_time)
# x = tsp.sample_most_likely(result.eigenstate)
# print("feasible:", qubo.is_feasible(x))
# z = tsp.interpret(x)
# print("solution:", z)
# print("solution objective:", tsp.tsp_value(z, adj_matrix))
# draw_tsp_solution(tsp.graph, z, colors, pos)
#
#
# algorithm_globals.random_seed = 123
# seed = 10598
#
# # create minimum eigen optimizer based on SamplingVQE
# vqe_optimizer = MinimumEigenOptimizer(vqe)
#
# # solve quadratic program
# result = vqe_optimizer.solve(qp)
# print(result.prettyprint())
#
# z = tsp.interpret(x)
# print("solution:", z)
# print("solution objective:", tsp.tsp_value(z, adj_matrix))
# draw_tsp_solution(tsp.graph, z, colors, pos)



"""
Links úteis: 
Sobre a biblioteca 'networkx':
    https://networkx.org/documentation/stable/reference/classes/digraph.html#networkx.DiGraph

Sobre os grafos:
    https://www.inf.ufsc.br/grafos/definicoes/definicao.html
    
Sobre as n-dimensional arrays do numpy:
    https://www.datacamp.com/tutorial/python-numpy-tutorial?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720824&utm_adgroupid=143216588537&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=665485585140&utm_targetid=dsa-1947282172981&utm_loc_interest_ms=&utm_loc_physical_ms=1031885&utm_content=dsa~page~community-tuto&utm_campaign=230119_1-sea~dsa~tutorials_2-b2c_3-row-p2_4-prc_5-na_6-na_7-le_8-pdsh-go_9-na_10-na_11-na-ltsjul23&gclid=Cj0KCQjwoK2mBhDzARIsADGbjepNehfa2bhNQZvl4Lcvk98eT6S8M6ly-KEjAqwL1jE1rcm5UBVCXP8aAnH7EALw_wcB 

Sobre o funcionamento do Perceptron:
    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj4jfuVicaAAxURpZUCHcz3CwQQFnoECBAQAw&url=https%3A%2F%2Fwww.simplilearn.com%2Ftutorials%2Fdeep-learning-tutorial%2Fperceptron%23%3A~%3Atext%3DIn%2520Perceptron%252C%2520the%2520weight%2520coefficient%2Cis%2520more%2520significant%2520than%2520zero.&usg=AOvVaw2Eph82bOMSi1JSFGZKt2RM&opi=89978449
"""
