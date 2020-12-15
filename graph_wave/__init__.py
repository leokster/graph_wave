import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import entropy
from scipy.sparse.linalg import eigs
import math
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class GraphWave:
    #define the heat kernel
    heat_kernel = lambda self, x, s: np.exp(-x * s)

    def __init__(self,
                 graph: nx.Graph,
                 weight_fun=None,
                 s=None,
                 degree=8,
                 pos=None):
        '''
        :param graph: an undirected nx.Grpah object
        :param weight_fun: callable maps edge value to numeric, which is then used as weight.
        :param full_computation:
        :param s:
        :param method:
        :param degree:
        '''
        self.s = s
        self.graph = graph.to_undirected()
        logger.info("created graph")
        self.__apply_weight_fun(weight_fun)
        logger.info("weight_fun applied")
        self.laplacian = self.__compute_laplacian()
        logger.info("computed laplacian")
        self.unit_matrix = self.__get_unit_matrix()
        logger.info("computed unit matrix")

        self.linear_wave_operator = self.get_fast_operator(degree=degree)
        logger.info("computed linear wave operator")

        if pos is None:
            self.__get_graph_position()
            logger.info("computed graph position")
        else:
            self.pos = pos

        self.graph_weights = self.__get_weights()
        logger.info("computed graph weights")


    def get_fast_operator(self, degree=20):
        '''
        Computes the solution of the differnetial equation the resulting n x n matrix gives the linear operator H on the graph which defines the
        evolution of heat activation on the nodes. Example: Assume x_0 is the activation at time 0 then the activation at time 1 is given by
        x_1 = H * x_0 (* is here the dot product).
        :return:
        '''

        max_eig = np.real(eigs(self.laplacian, k=1, return_eigenvectors=False))[0]
        coeffs = np.real(np.polynomial.chebyshev.Chebyshev.interpolate(
            lambda x: self.heat_kernel(x, self.s), domain=[0, max_eig], deg=degree
        ).coef.astype("float32"))

        alpha = max_eig / 2
        v_m_2 = self.unit_matrix
        v_m_1 = (self.laplacian - alpha * self.unit_matrix) / alpha

        cheby_approx = coeffs[0] * v_m_2 + coeffs[1] * v_m_1
        for i in range(2, degree):
            v_m_0 = 2 / alpha * self.laplacian @ v_m_1 - 2 * v_m_1 - v_m_2
            cheby_approx = cheby_approx + coeffs[i] * v_m_0
            v_m_2 = v_m_1
            v_m_1 = v_m_0

        return cheby_approx

    def __set_random_weights(self):
        '''
        Method sets the weights of the graph object to random weights. Should not be used
        :return:
        '''
        for (u, v) in self.graph.edges():
            self.graph.edges[u, v]['weight'] = random.randint(1, 10)

    def __apply_weight_fun(self, weight_fun):
        if weight_fun == None:
            return None
        else:
            for u, v, c in self.graph.edges.data():
                self.graph.edges[u, v]["weight"] = weight_fun(c)

    def __get_weights(self):
        '''
        Returns The weights of the current graph (stored in self.graph)
        :return:
        '''
        try:
            return {(i, j): self.laplacian[i, j] * -1 for i, j in self.graph.edges()}
        except IndexError:
            return nx.get_edge_attributes(self.graph, "weight")

    def __compute_laplacian(self):
        '''
        Computes the graph Laplacian
        :return:
        '''
        return nx.laplacian_matrix(self.graph).toarray().real.astype("float32")

    def __eigenvector_decomposition(self):
        '''
        Computes Eigenvector decomposition
        :return:
        '''
        return np.linalg.eig(self.laplacian)

    def __get_graph_position(self):
        '''
        Computes the position of the graph nodes (only used for visualization)
        :return:
        '''
        lon = nx.get_node_attributes(self.graph, "longitude")
        lat = nx.get_node_attributes(self.graph, "latitude")
        if (len(lon) == 0):
            self.pos = nx.spring_layout(self.graph)
        else:
            self.pos = {node: (lo, lat[node]) for node, lo in lon.items()}
            self.nan_position = [a for a, b in self.pos.items() if b[0] == "unknown" or math.isnan(b[0])]
            fixed_position = [a for a in self.graph.nodes() if a not in self.nan_position]
            for a in self.nan_position:
                random_angle = np.exp(1j * random.uniform(0, 2 * math.pi))
                self.pos[a] = (8.5 + random_angle.real * 3.5, 46.8 + random_angle.imag * 2)
            self.pos = nx.spring_layout(self.graph, k=0.1, pos=self.pos, fixed=fixed_position)


    def __get_unit_matrix(self):
        '''
        Returns just a simple unit matrix with ones on the diagonal of the size of the Laplacian
        :return:
        '''
        return np.diag([1] * self.graph.number_of_nodes())

    def get_heat_activation(self, start_time: int = 0, end_time: int = 0, initial_condition=0):
        '''
        Computes a timeline of node activations following the heat equation.
        :param start_time: (int): The start time 0 would be starting with the initial condition
        :param end_time: (int): The end time. If end_time =< start_time, then there will only be one snapshot of time start_time.
        :param initial_condition: (list/int): an explicit activation of the graph as numeric list of all nodes or an
        integer indicating a node identity which is activated with the delta function (i.e. 1 on this node and 0 everywhere else)
        :return:
        '''
        if (not isinstance(initial_condition, list) and not isinstance(initial_condition, np.ndarray)):
            initial_condition = self.unit_matrix[:, initial_condition]
        time_series = []

        time_series = time_series + [
            np.linalg.matrix_power(self.linear_wave_operator, start_time).dot(initial_condition).real]
        duration = max(end_time - start_time, 0)
        for i in range(duration):
            time_series = time_series + [self.linear_wave_operator.dot(time_series[-1]).real]

        return time_series

    def __modified_entropy(self, pk):
        '''
        slightly modified information criterion entropy
        :param pk:
        :return:
        '''
        return entropy(pk=np.maximum(pk, 0), base=2)

    def get_entropy_embedding(self, length: int = 10):
        '''
        computes an entropy embedding of length timesteps. I.e. for each node the heat activation of the entire graph will be
        computed if the specific node is activated with the dirac-delta function. By considering the activation of the nodes
        in the graph at each timestap as discrete probability distribution, one can compute the entropy of this distribution
        at each timestep. Hence we get for each node an embedding in a length-dimensional vector space. This embedding is returened here.
        :param length: size of the embedding
        :return:
        '''
        self.entropy_embedding = {i: [self.__modified_entropy(d) for d in self.get_heat_activation(0, length, i)] for i
                                  in range(self.laplacian.shape[0])}
        return self.entropy_embedding


    def __characteristic_function(self, pk: list, t: float) -> list:
        '''
        computes the characteristic function at time t
        :param pk:
        :return:
        '''
        char_fun_val = np.sum(np.exp(1j * t * np.maximum(pk, 0)))
        return [char_fun_val.real, char_fun_val.imag]

    def __characteristic_function_range(self, pk, length):
        return [comp for t in range(length) for comp in self.__characteristic_function(pk, t)]

    def get_char_fun_embedding(self, length: int = 10):
        '''
        computes an embedding based on the characteristic function of the discrete probability distribution given by the heat
        activation of the graph nodes (read get_entropy_embedding for detailed information)
        :param length: size of the embedding
        :return:
        '''
        self.char_fun_embedding = {i: self.__characteristic_function_range(self.get_heat_activation(1, 1, i), length)
                                   for i in range(self.laplacian.shape[0])}
        return self.char_fun_embedding

    def __update_plot(self, num, activation_time_series, ax, show_edge_label=False, fixed_v_range=False):
        idx = num % len(activation_time_series)
        ax.clear()
        ax.set_title("Frame %d: sum: %s    min: %s    max: %s" % (
        idx + 1, np.round(sum(activation_time_series[idx]), 3), np.round(min(activation_time_series[idx]), 3),
        np.round(max(activation_time_series[idx]), 3)), fontweight="bold")
        if fixed_v_range:
            vmin = 0
            vmax = 1
        else:
            vmin = vmax = None
        nx.draw(self.graph, self.pos, node_color=activation_time_series[idx], cmap=plt.cm.Reds, ax=ax, vmin=vmin,
                vmax=vmax, alpha=0.9, node_size=50)
        if show_edge_label: nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=self.graph_weights, ax=ax)

    def visualize_propagation(self, start_time: int = 0, end_time: int = None, initial_condition=0,
                              save_path: str = None, show_edge_label=False, fixed_v_range=False):
        activation_time_series = self.get_heat_activation(start_time, end_time, initial_condition)

        fig, ax = plt.subplots()
        frames = len(activation_time_series)
        ani = FuncAnimation(fig,
                            lambda num: self.__update_plot(num,
                                                           activation_time_series,
                                                           ax,
                                                           show_edge_label=show_edge_label,
                                                           fixed_v_range=fixed_v_range),
                            frames=frames,
                            interval=1,
                            repeat=True)
        if save_path is not None:
            ani.save(save_path, writer='imagemagick', fps=min(frames / 2, 30))
            return True
        else:
            return ani


if __name__ == "__main__":
    g1 = nx.karate_club_graph()
    for (u, v) in g1.edges():
        g1.edges[u, v]['weight'] = random.randint(1, 10)

    gw = GraphWave(g1, s=0.001)
    gw.get_fast_operator(8)
    fig = gw.visualize_propagation(start_time=0, end_time=100, initial_condition=0, fixed_v_range=True, save_path="output.gif")
    gw.get_entropy_embedding()