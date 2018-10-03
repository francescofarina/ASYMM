import numpy as np
from copy import deepcopy


def is_pos_def(matrix):
    """
    :param matrix: a matrix
    :return: true if the matrix is positive definite
    """
    return np.all(np.linalg.eigvals(matrix) > 0)


class Node:

    def __init__(self, n, graph_diameter, name, neighbors):
        """
        :param n: dimension of optimization variable (x\in R^n)
        :param graph_diameter: diameter of the graph
        :param name: name of the node
        :param neighbors: neighbors of the node
        """
        self.n = n
        self.neighbors = neighbors
        self.name = name
        self.graph_diameter = graph_diameter

        # logic-AND matrix
        self.S = np.zeros([graph_diameter, len(neighbors) + 1])
        self.S_ind = {}
        for j in range(len(neighbors)):
            self.S_ind[neighbors[j]] = j

        # optimization variable
        self.x = np.zeros([n, 1])
        self.x_sequence = np.zeros([n, 1])
        self.x_sequence_in = np.zeros([n, 1])

        # auxiliary variables
        self.local_dual_update = False # M_done in the paper
        self.counter = 0

        # Algorithm parameters
        # max penalty
        self.rho_max = 500000
        # constant stepsize
        self.alpha = 0.001
        # initial tolerance
        self.e = 5
        # penalty threshold
        self.gamma = 0.25
        # penalty growth parameter
        self.beta = 4

        # Cost function
        self.objective_function = None
        self.objective_gradient = None
        self.objective_hessian = None

        # Constraints
        self.equality_constraints = []
        self.equality_gradients = []
        self.equality_hessians = []
        self.lambdas = []
        self.vrho = []

        self.inequality_constraints = []
        self.inequality_gradients = []
        self.inequality_hessians = []
        self.mus = []
        self.zeta = []

        # Multipliers and neighbors estimates
        self.nu_to = {}
        self.rho_ij = {}
        self.nu_from = {}
        self.nu_from_old = {}
        self.rho_ji = {}
        self.x_neighbors = {}
        self.x_neighbors_old = {}

        for i in self.neighbors:
            self.nu_to[i] = 0.01*np.random.rand(n,1) # np.zeros([n, 1])
            self.nu_from[i] = 0.01*np.random.rand(n,1) # np.zeros([n, 1])
            self.nu_from_old[i] = 0.01*np.random.rand(n,1) # np.zeros([n, 1])
            self.rho_ij[i] = 1
            self.rho_ji[i] = 1
            self.x_neighbors[i] = np.zeros([n, 1])
            self.x_neighbors_old[i] = np.zeros([n, 1])

    def __str__(self):
        description = "Node {}, neighbor of nodes".format(self.name)
        for i in self.neighbors:
            description += " "+str(i)
        description += ".\n"
        description += "x = {}\n".format(self.x.flatten())
        return description

    def set_initial_value(self, x):
        """
        :param x: initial estimate
        """
        self.x += x
        self.x_sequence += x

    def add_objective_function(self, fun, gradient, hessian=None):
        """
        :param fun: objective function
        :param gradient: objective function gradient
        :param hessian: objective function hessian (optional)
        """
        self.objective_function = fun
        self.objective_gradient = gradient
        self.objective_hessian = hessian

    def add_equality_constraint(self, fun, gradient, hessian=None):
        """
        :param fun: equality constraint function h(x) = 0
        :param gradient: equality constraint gradient
        :param hessian: equality constraint hessian (optional)
        """
        self.equality_constraints.append(fun)
        self.equality_gradients.append(gradient)
        self.equality_hessians.append(hessian)
        # initialize corresponding multiplier and penalty parameter
        self.lambdas.append(0.01*np.random.rand())
        self.vrho.append(1)

    def add_inequality_constraint(self, fun, gradient, hessian=None):
        """
        :param fun: inequality constraint function g(x) <= 0
        :param gradient: equality constraint gradient
        :param hessian: equality constraint hessian (optional)
        """
        self.inequality_constraints.append(fun)
        self.inequality_gradients.append(gradient)
        self.inequality_hessians.append(hessian)
        # initialize corresponding multiplier and penalty parameter
        self.mus.append(0.01*np.random.rand()) 
        self.zeta.append(1)

    def get_x_from_neighbor(self, neighbor, x_new):
        """
        :param neighbor: neighbor from which information is received
        :param x_new: new value of x (of the sending neighbor)
        """
        self.x_neighbors[neighbor] = x_new

    def get_nu_from_neighbor(self, neighbor, nu_new):
        """
        :param neighbor: neighbor from which information is received
        :param nu_new: new value of multiplier nu (of the sending neighbor)
        """
        self.nu_from[neighbor] = nu_new

    def get_rho_from_neighbor(self, neighbor, rho_new):
        """
        :param neighbor: neighbor from which information is received
        :param rho_new: new value of penalty rho (of the sending neighbor)
        """
        self.rho_ji[neighbor] = rho_new

    def local_lagrangian(self, point=None):
        """
        :param point: optional
        :return: value of the local augmented lagrangian at a given point or at self.x
        """
        if point is None:
            point = deepcopy(self.x)
        result = 0

        # objective function
        result += float(self.objective_function(point))

        # neighboring constraints
        for j in self.neighbors:
            result += point.T.dot(self.nu_to[j] - self.nu_from[j]) +\
                      (self.rho_ij[j]+self.rho_ji[j]) / 2 * np.linalg.norm(point - self.x_neighbors[j]) ** 2

        # equality constraints
        for i in range(len(self.equality_constraints)):
            result += self.lambdas[i]*(self.equality_constraints[i](point)) + \
                      (self.vrho[i]*np.linalg.norm(self.equality_constraints[i](point))**2)/2

        # inequality constraints
        for i in range(len(self.inequality_constraints)):
            result += 1/(2*self.zeta[i]) * \
                      (max(0, self.mus[i] + self.zeta[i]*self.inequality_constraints[i](point))**2 - self.mus[i]**2)

        return float(result)

    def local_lagrangian_gradient(self, point=None):
        """
        :param point: optional
        :return: value of the local augmented lagrangian gradient at a given point or at self.x
        """
        if point is None:
            point = deepcopy(self.x)
        result = np.zeros([self.n, 1])

        # objective function gradient
        result += self.objective_gradient(point)
        # neighboring constraints
        for j in self.neighbors:
            result += (self.nu_to[j] - self.nu_from[j]) + \
                      (self.rho_ij[j] + self.rho_ji[j])*(point - self.x_neighbors[j])

        # equality constraints
        for i in range(len(self.equality_constraints)):
            result += self.lambdas[i]*self.equality_gradients[i](point) +\
                      self.vrho[i]*self.equality_constraints[i](point)*self.equality_gradients[i](point)

        # inequality constraints
        for i in range(len(self.inequality_constraints)):
            if self.inequality_constraints[i](point) > -self.mus[i]/self.zeta[i]:
                result += (self.mus[i] + self.zeta[i]*self.inequality_constraints[i](point)) *\
                          self.inequality_gradients[i](point)

        return result

    def local_lagrangian_hessian(self, point=None):
        """
        :param point: optional
        :return: value of the local augmented lagrangian hessian at a given point or at self.x
        """
        if point is None:
            point = deepcopy(self.x)
        result = np.zeros([self.n, self.n])

        result += self.objective_hessian(point)

        # neighboring constraints
        for j in self.neighbors:
            result += (self.rho_ij[j]+self.rho_ji[j])*np.eye(self.n)

        # equality constraints
        for i in range(len(self.equality_constraints)):
            p1 = self.equality_hessians[i](point)*self.equality_constraints[i](point)
            p2 = self.equality_gradients[i](point).dot(self.equality_gradients[i](point).T)
            result += self.lambdas[i]*self.equality_hessians[i](point) + self.vrho[i]*(p1+p2)

        # inequality constraints
        for i in range(len(self.inequality_constraints)):
            if self.inequality_constraints[i](point) > -self.mus[i] / self.zeta[i]:
                p1 = self.inequality_hessians[i](point) * self.inequality_constraints[i](point)
                p2 = self.inequality_gradients[i](point).dot(self.inequality_gradients[i](point).T)
                result += self.mus[i]*self.inequality_hessians[i](point) + self.zeta[i]*(p1+p2)

        return result

    def perform_gradient_descent(self, step_type="Newton"):
        """
        Perform a gradient descent step on the local augmented Lagrangian
        :param step_type: "constant" uses a constant stepsize self.alpha, "Newton" approximate the Lipschitz constant through the Hessian
        """
        if step_type == "constant":
            # Constant step
            gradient = self.local_lagrangian_gradient()
            self.x -= self.alpha * gradient

        elif step_type == "Newton":
            # Newton
            gradient = self.local_lagrangian_gradient()
            hessian = self.local_lagrangian_hessian()
            if is_pos_def(hessian):
                self.x -= 0.5 * np.linalg.inv(hessian).dot(gradient)
            else:
                self.x -= min(self.alpha, self.e/10) * gradient

    def local_matrix_update(self):
        """
        Update the matrix S
        """
        criterion = np.linalg.norm(self.local_lagrangian_gradient()) <= self.e
        self.S[0, -1] = int(criterion)
        for l in range(self.graph_diameter-1):
            self.S[l+1, -1] = np.prod(self.S[l, :])

    def force_matrix_update(self):
        """
        Force the matrix S update (to be used when receiving a new multiplier)
        """
        self.S[-1] = 1
        # self.S[:-1] = 0

    def triggered_matrix_update(self, neighbor, column):
        """
        Update the matrix when receiving information from a neighbor
        :param neighbor: neighbor sending information
        :param column: column to update
        :return:
        """
        not_updated = 0
        for j in self.neighbors:
            not_updated += np.array_equal(self.nu_from[j], self.nu_from_old[j])
        if not_updated == len(self.neighbors):
            index = self.S_ind[neighbor]
            self.S[0, index] = column[0]
            for l in range(1, self.graph_diameter):
                self.S[l, index] = column[l]

    # Algorithm steps
    def primal_update_step(self):
        """
        Perform a primal descent step and a local matrix update
        """
        # self.x_sequence_in = np.append(self.x_sequence_in, self.x, axis=1)
        self.perform_gradient_descent()
        self.local_matrix_update()

    def dual_update_step(self):
        """
        Updates multipliers and penalty parameters
        """
        # self.counter = 0
        x_old = deepcopy(self.x_sequence[:, -1]).reshape(self.n, 1)
        self.x_sequence = np.append(self.x_sequence, self.x, axis=1)
        # coupling duals
        for j in self.neighbors:
            self.nu_to[j] += self.rho_ij[j] * (self.x - self.x_neighbors[j])
            if np.linalg.norm(self.x - self.x_neighbors[j]) > self.gamma * \
                    np.linalg.norm(x_old - self.x_neighbors_old[j]):
                self.rho_ij[j] = min(self.beta*self.rho_ij[j], self.rho_max)
            neigh_old = deepcopy(self.x_neighbors[j])
            self.x_neighbors_old[j] = neigh_old

        # equality duals
        for i in range(len(self.equality_constraints)):
            self.lambdas[i] += self.vrho[i]*self.equality_constraints[i](self.x)
            if np.linalg.norm(self.equality_constraints[i](self.x)) > self.gamma * \
                    np.linalg.norm(self.equality_constraints[i](x_old)):
                self.vrho[i] = min(self.beta * self.vrho[i], self.rho_max)

        # inequality duals
        for i in range(len(self.inequality_constraints)):
            mu_old = self.mus[i]
            self.mus[i] = max(0, self.mus[i] + self.zeta[i]*self.inequality_constraints[i](self.x))
            if np.linalg.norm(max(-self.mus[i]/self.zeta[i], self.inequality_constraints[i](self.x))) > self.gamma * \
                    np.linalg.norm(max(-mu_old/self.zeta[i], self.inequality_constraints[i](x_old))):
                self.zeta[i] = min(self.beta * self.zeta[i], self.rho_max)

        self.local_dual_update = True

    def reset_step(self):
        """
        Reset the matrix S and update e
        """
        self.local_dual_update = False
        self.S = np.zeros([self.graph_diameter, len(self.neighbors) + 1])
        for j in self.neighbors:
            nu_to_pass = deepcopy(self.nu_from[j])
            self.nu_from_old[j] = nu_to_pass
        self.e = self.e/1.2

