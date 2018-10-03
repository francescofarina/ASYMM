import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.ticker import MaxNLocator

from node_definition import Node


def gen_color():
    """
    :return: a random color
    """
    return "#%02x%02x%02x" % (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))


# Set the seed for the simulation
np.random.seed(128)

f_size = 15
figure_size = (6, 6)

# Simulation setup
# number of awakenings
iterations = 10000
# dimension of the optimization variable
n = 2
# number of nodes
N = 10

# colors to be used in plots
col_num = len(plt.rcParams['axes.prop_cycle'])-1
colors = []
for i, color in enumerate(plt.rcParams['axes.prop_cycle']):
    colors.append(color['color'])
if N > col_num:
    for i in range(col_num, N):
        colors.append(gen_color())

# Graph generation
print("Generating graph...")
while True:
    p = 0.2
    Adj = np.random.binomial(1, p, (N, N))  # each entry is 1 with prob "p"

    Adj = np.logical_or(Adj, Adj.transpose()) # Undirected
    I_NN = np.eye(N)

    Adj = np.logical_and(Adj, np.logical_not(I_NN)).astype(int)  # remove self - loops
    testAdj = np.linalg.matrix_power((I_NN + Adj), N)  # check if G is connected

    check = 0
    for i in range(N):
        check += int(bool(len(np.nonzero(Adj[:, i])[0]))) + int(bool(len(np.nonzero(Adj[i])[0])))

    if not np.any(np.any(np.logical_not(testAdj))) and check == 2*N:
        break


G = nx.Graph(Adj)
diameter = nx.diameter(G)
# A = nx.to_numpy_matrix(G)
# Adj = np.asarray(A.astype(int))

# nodes creation
print("Initializing nodes...")
nodes = {}
for i in range(N):
    nodes[i] = Node(n, diameter, i, np.nonzero(Adj[:, i])[0])


# node positions
C = np.zeros([n, N])

# true position of the source
x_true = 5 * (np.random.rand(n, 1) - 0.5)

# measures
if n == 2:
    fig1 = plt.figure()

epsilon = np.zeros([N, 1])
ineq1 = {}
ineq1_grad = {}
ineq1_hess = {}
ineq2 = {}
ineq2_grad = {}
ineq2_hess = {}
c = {}
pos = {}
a = {}
b = {}
w = {}
y = {}
for i in range(N):
    # noise standard deviation
    w[i] = 0.3 * np.random.rand()
    # node position
    c[i] = 5 * (np.random.rand(n, 1) - 0.5)
    pos[i] = c[i].flatten()
    # measured distance from source
    y[i] = np.linalg.norm(c[i] - x_true) + w[i] * (np.random.rand() - 0.5)
    # bounds
    b[i] = y[i] + w[i]
    if y[i] > w[i]:
        a[i] = y[i] - w[i]
    else:
        a[i] = 0

    # plot feasible region
    if n == 2:
        cir = np.linspace(-b[i], b[i], 1000, endpoint=True)
        outer = b[i] * np.sin(np.arccos(cir / b[i]))  # x-axis values -> outer circle
        inner = a[i] * np.sin(np.arccos(cir / a[i]))  # x-axis values -> inner circle (with nan's beyond circle)
        inner[np.isnan(inner)] = 0.  # inner now looks like a boulder hat, meeting outer at the outer points

        plt.fill_between(cir+c[i][0], inner+c[i][1], outer+c[i][1], facecolor=colors[i], alpha=0.3)
        plt.fill_between(cir+c[i][0], -outer+c[i][1], -inner+c[i][1], facecolor=colors[i], alpha=0.3)

    # initialize node (x^0)
    nodes[i].set_initial_value(10 * (np.random.rand(n, 1) - 0.5))

    # objective function
    # # # Linear
    # objective_function = lambda x: x[0]
    # vector = np.zeros([n, 1])
    # vector[0] = 1
    # objective_gradient = lambda x: vector
    # objective_hessian = lambda x: np.zeros([n,n])
    # nodes[i].add_objective_function(objective_function, objective_gradient, objective_hessian)

    # # Quadratic
    objective_function = lambda x: float(x.T.dot(x))
    objective_gradient = lambda x: x
    objective_hessian = lambda x: np.eye(n)
    nodes[i].add_objective_function(objective_function, objective_gradient, objective_hessian)

    # inequality 1: outer border
    ineq1[i] = lambda x: np.linalg.norm(x - c[i]) ** 2 - b[i] **2
    ineq1_grad[i] = lambda x: 2 * (x - c[i])
    ineq1_hess[i] = lambda x: 2 * np.eye(n)
    nodes[i].add_inequality_constraint(ineq1[i], ineq1_grad[i], ineq1_hess[i])

    # inequality 2: inner border
    ineq2[i] = lambda x: a[i] ** 2 - np.linalg.norm(x - c[i]) ** 2
    ineq2_grad[i] = lambda x: - 2 * (x - c[i])
    ineq2_hess[i] = lambda x: -2 * np.eye(n)
    nodes[i].add_inequality_constraint(ineq2[i], ineq2_grad[i], ineq2_hess[i])

    # pass initial estimate to neighbors
    for j in nodes[i].neighbors:
        nodes[j].get_x_from_neighbor(i, nodes[i].x)
        nodes[j].nu_from[i] = deepcopy(nodes[i].nu_to[j])
        nodes[j].nu_from_old[i] = deepcopy(nodes[i].nu_to[j])

# Variables for storing the sequence of x_i
sequence_in = {}
for i in range(N):
    sequence_in[i] = np.zeros([n, iterations])
    sequence_in[i][:, 0] = nodes[i].x.flatten()

# counter of unused awakenings (node is awake, but do nothing)
unused_iterations = 0

# ASYMM
print("\nASYMM start")
for k in range(iterations):
    i = np.random.randint(N)
    print("Iteration {}, Node {} AWAKE".format(k, i), end="\r")

    # TASK T1
    if np.prod(nodes[i].S[-1, :]) != 1 and not nodes[i].local_dual_update:
        nodes[i].primal_update_step()
        for j in nodes[i].neighbors:
            nodes[j].get_x_from_neighbor(i, nodes[i].x)
            nodes[j].triggered_matrix_update(i, nodes[i].S[:, -1])

    for j in range(N):
        sequence_in[j][:, k] = nodes[j].x.flatten()

    # TASK T2
    if np.prod(nodes[i].S[-1, :]) == 1 and not nodes[i].local_dual_update:
        print("Node {} updating duals".format(i), end="\r")
        nodes[i].dual_update_step()

        for j in nodes[i].neighbors:
            nodes[j].get_nu_from_neighbor(i, nodes[i].nu_to[j])
            nodes[j].get_rho_from_neighbor(i, nodes[i].rho_ij[j])
            nodes[j].force_matrix_update()

    if np.prod(nodes[i].S[-1, :]) == 1 and nodes[i].local_dual_update:
        unused_iterations += 1

    not_updated = 0
    for j in nodes[i].neighbors:
        not_updated += np.array_equal(nodes[i].nu_from[j], nodes[i].nu_from_old[j])
    all_updated = not not_updated

    if nodes[i].local_dual_update and all_updated:
        print("Node {} reset".format(i), end="\r")
        nodes[i].reset_step()
print("\nASYMM end")

print("\nNodes have been awake without doing anything {} times over {} total awakenings".format(unused_iterations, iterations))
for i in range(N):
    np.append(nodes[i].x_sequence, nodes[i].x, axis=1)

# plot nodes location, feasible regions (if n=2) and x_i^k (if n=2)
if n == 2:
    for i in range(N):
        plt.plot(nodes[i].x_sequence[0, :], nodes[i].x_sequence[1, :], colors[i])
    plt.axis('equal')

nx.draw_networkx_nodes(G, pos,
                       node_color=colors,
                       node_size=300,
                       alpha=0.5)
nx.draw_networkx_edges(G, pos,
                       alpha=0.8)
labels = {}
for i in range(N):
    labels[i] = r'$%d$' % (i)
nx.draw_networkx_labels(G, pos, labels, font_size=10)
plt.show()

fig2, ax2 = plt.subplots(figsize=figure_size)
for i in range(N):
    for l in range(n):
        ax2.plot(nodes[i].x_sequence[l, :], color=colors[i])
ax2.set_xlabel(r"cycle $k$", fontsize=f_size)
ax2.set_ylabel(r"$x_i^k$", fontsize=f_size)
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.tick_params(labelsize=f_size)

#
estimates = len(nodes[i].x_sequence[0, :])
obj_values = np.zeros([estimates, 1])
infeasibilty_measures = np.zeros([estimates, 1])
for k in range(estimates):
    for i in range(N):
        obj_values[k] += nodes[i].objective_function(nodes[i].x_sequence[:, k].reshape(n, 1))
        for neigh in nodes[i].neighbors:
            infeasibilty_measures[k] += np.linalg.norm(nodes[i].x_sequence[:, k].reshape(n, 1) - nodes[neigh].x_sequence[:, k].reshape(n, 1))
        for ineq in nodes[i].inequality_constraints:
            infeasibilty_measures[k] += max(0, ineq(nodes[i].x_sequence[:, k].reshape(n, 1)))

fig4, ax4 = plt.subplots(figsize=figure_size)
ax4.set_yscale("log", nonposy='clip')
ax4.set_ylim(ymin=np.min(infeasibilty_measures), ymax=np.max(infeasibilty_measures))
ax4.plot(infeasibilty_measures)
for tick in ax4.get_xticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(f_size)
for tick in ax4.get_yticklabels():
    tick.set_fontname("Times New Roman")
    tick.set_fontsize(f_size)
ax4.set_xlabel(r"cycle $k$", fontsize=f_size)
ax4.set_ylabel(r"$\log(\xi^k)$", fontsize=f_size)
ax4.xaxis.set_major_locator(MaxNLocator(integer=True))


fig222 = plt.figure(figsize=figure_size)
for i in range(N):
    for l in range(n):
        plt.plot(sequence_in[i][l, :], color=colors[i])
plt.ylim(-3, 3)
plt.xlim(0, 10000)
plt.xticks(fontsize=f_size)
plt.yticks(fontsize=f_size)
plt.xlabel(r"iteration", fontsize=f_size)
plt.ylabel(r"$x_i^t$", fontsize=f_size)

plt.show()

