"""Custom graph utilities for generating diverse graph families."""

import networkx as nx


class CustomGraphPuzzle:
    """Custom graph for coloring."""

    def __init__(self, graph=None, name="Custom Graph"):
        self.name = name
        self.graph = graph if graph is not None else nx.Graph()

    @property
    def positions(self):
        return nx.get_node_attributes(self.graph, "pos")

    def compute_layout(self, layout="spring", seed=None):
        """Attach layout coordinates to nodes for consistent visualization."""
        if layout == "circular":
            pos = nx.circular_layout(self.graph)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(self.graph)
        elif layout == "shell":
            pos = nx.shell_layout(self.graph)
        else:
            pos = nx.spring_layout(self.graph, seed=seed)

        nx.set_node_attributes(self.graph, pos, "pos")
        return pos

    def to_graph(self):
        """Return the graph representation."""
        return self.graph

    def to_dict(self):
        return {
            "name": self.name,
            "nodes": list(self.graph.nodes()),
            "edges": [[u, v] for u, v in self.graph.edges()],
            "coordinates": {k: [float(v[0]), float(v[1])] for k, v in self.positions.items()},
        }

    def add_node(self, node_id):
        self.graph.add_node(node_id)

    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    def remove_node(self, node_id):
        self.graph.remove_node(node_id)

    def remove_edge(self, node1, node2):
        self.graph.remove_edge(node1, node2)

    def visualize_plotly(self, coloring=None, title="Custom Graph", positions=None):
        from utils.visualization import visualize_graph_plotly

        positions = positions or self.positions
        return visualize_graph_plotly(self.graph, coloring=coloring, title=title, positions=positions)

    @staticmethod
    def create_complete_graph(n):
        return CustomGraphPuzzle(nx.complete_graph(n), name=f"Complete Graph K{n}")

    @staticmethod
    def create_cycle_graph(n):
        return CustomGraphPuzzle(nx.cycle_graph(n), name=f"Cycle Graph C{n}")

    @staticmethod
    def create_wheel_graph(n):
        return CustomGraphPuzzle(nx.wheel_graph(n), name=f"Wheel Graph W{n}")

    @staticmethod
    def create_bipartite_graph(n1, n2, p=0.3, seed=None):
        graph = nx.bipartite.random_graph(n1, n2, p, seed=seed)
        return CustomGraphPuzzle(graph, name=f"Bipartite Graph B({n1},{n2})")

    @staticmethod
    def create_random_graph(n, p, seed=None):
        graph = nx.erdos_renyi_graph(n, p, seed=seed)
        return CustomGraphPuzzle(graph, name="Random Erdős–Rényi")

    @staticmethod
    def create_petersen_graph():
        return CustomGraphPuzzle(nx.petersen_graph(), name="Petersen Graph")
