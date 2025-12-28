"""Map coloring puzzle utilities with planar and geographic generation."""

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay


class MapColoringPuzzle:
    """Map coloring puzzle represented as a graph coloring problem."""

    def __init__(self, regions=None, coordinates=None, name="Map Coloring"):
        """Initialize a map coloring puzzle."""
        self.name = name
        self.regions = self._normalize_regions(regions or {})
        self.coordinates = coordinates or {}
        self.graph = self._create_graph()

    def _normalize_regions(self, regions):
        normalized = {}
        for region, neighbors in regions.items():
            normalized.setdefault(region, set())
            for neighbor in neighbors:
                if neighbor == region:
                    continue
                normalized[region].add(neighbor)
                normalized.setdefault(neighbor, set()).add(region)
        return {k: sorted(v) for k, v in normalized.items()}

    def _create_graph(self):
        graph = nx.Graph()
        for region in self.regions.keys():
            graph.add_node(region)
        for region, neighbors in self.regions.items():
            for neighbor in neighbors:
                graph.add_edge(region, neighbor)
        if self.coordinates:
            nx.set_node_attributes(graph, self.coordinates, "pos")
        return graph

    @property
    def positions(self):
        return nx.get_node_attributes(self.graph, "pos")

    def to_graph(self):
        """Return the graph representation."""
        return self.graph

    def to_dict(self):
        """Serialize puzzle to a lightweight dictionary for persistence."""
        return {
            "name": self.name,
            "regions": self.regions,
            "coordinates": {k: [float(v[0]), float(v[1])] for k, v in self.coordinates.items()},
            "nodes": list(self.graph.nodes()),
            "edges": [[u, v] for u, v in self.graph.edges()],
        }

    def visualize_plotly(self, coloring=None, title="Map Coloring"):
        from utils.visualization import visualize_graph_plotly

        return visualize_graph_plotly(self.graph, coloring=coloring, title=title, positions=self.positions)

    @staticmethod
    def create_random_planar(num_regions=12, seed=None, spread=1.0):
        """Create a random planar map using Delaunay triangulation of random points."""
        rng = np.random.default_rng(seed)
        points = rng.random((num_regions, 2)) * spread
        triangulation = Delaunay(points)

        edges = set()
        for simplex in triangulation.simplices:
            simplex_edges = [(simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3)]
            for i, j in simplex_edges:
                a, b = sorted((i, j))
                edges.add((a, b))

        regions = {f"R{idx + 1}": [] for idx in range(num_regions)}
        coordinates = {f"R{idx + 1}": points[idx].tolist() for idx in range(num_regions)}

        for i, j in edges:
            ri, rj = f"R{i + 1}", f"R{j + 1}"
            regions[ri].append(rj)
            regions[rj].append(ri)

        return MapColoringPuzzle(regions, coordinates, name="Random Planar Map")

    @staticmethod
    def create_usa_map():
        """Create a simplified USA map coloring puzzle with coordinates."""
        usa_regions = {
            "WA": ["OR", "ID"],
            "OR": ["WA", "ID", "NV", "CA"],
            "CA": ["OR", "NV", "AZ"],
            "ID": ["WA", "OR", "NV", "UT", "WY", "MT"],
            "NV": ["OR", "CA", "AZ", "UT", "ID"],
            "UT": ["ID", "WY", "CO", "AZ", "NV"],
            "AZ": ["CA", "NV", "UT", "NM"],
            "MT": ["ID", "WY", "ND", "SD"],
            "WY": ["MT", "ID", "UT", "CO", "NE", "SD"],
            "CO": ["WY", "UT", "NM", "OK", "KS", "NE"],
            "NM": ["AZ", "UT", "CO", "OK", "TX"],
            "ND": ["MT", "SD"],
            "SD": ["ND", "MT", "WY", "NE"],
            "NE": ["SD", "WY", "CO", "KS"],
            "KS": ["NE", "CO", "OK"],
            "OK": ["KS", "CO", "NM", "TX"],
            "TX": ["OK", "NM"],
        }

        coords = {
            "WA": [-120.0, 47.5],
            "OR": [-120.5, 44.0],
            "CA": [-119.5, 36.5],
            "ID": [-114.0, 44.2],
            "NV": [-117.0, 39.5],
            "UT": [-111.8, 39.0],
            "AZ": [-111.5, 34.0],
            "MT": [-109.6, 47.0],
            "WY": [-107.5, 43.0],
            "CO": [-105.5, 39.0],
            "NM": [-106.0, 34.5],
            "ND": [-100.5, 47.6],
            "SD": [-99.9, 44.5],
            "NE": [-99.4, 41.5],
            "KS": [-98.2, 38.5],
            "OK": [-97.5, 35.5],
            "TX": [-99.0, 31.5],
        }

        return MapColoringPuzzle(usa_regions, coords, name="USA Map")

    @staticmethod
    def create_europe_map():
        """Create a simplified Europe map coloring puzzle with coordinates."""
        europe_regions = {
            "Portugal": ["Spain"],
            "Spain": ["Portugal", "France"],
            "France": ["Spain", "Belgium", "Germany", "Switzerland", "Italy"],
            "Belgium": ["France", "Netherlands", "Germany"],
            "Netherlands": ["Belgium", "Germany"],
            "Germany": ["Netherlands", "Belgium", "France", "Switzerland", "Austria", "Poland", "Czech Republic"],
            "Switzerland": ["France", "Germany", "Austria", "Italy"],
            "Austria": ["Germany", "Switzerland", "Italy", "Czech Republic", "Hungary"],
            "Italy": ["France", "Switzerland", "Austria"],
            "Poland": ["Germany", "Czech Republic"],
            "Czech Republic": ["Germany", "Poland", "Austria"],
            "Hungary": ["Austria"],
        }

        coords = {
            "Portugal": [-8.0, 39.5],
            "Spain": [-4.0, 40.0],
            "France": [2.5, 46.5],
            "Belgium": [4.5, 50.6],
            "Netherlands": [5.5, 52.2],
            "Germany": [10.0, 51.0],
            "Switzerland": [8.0, 46.8],
            "Austria": [14.0, 47.5],
            "Italy": [12.5, 42.8],
            "Poland": [19.0, 52.0],
            "Czech Republic": [15.5, 49.8],
            "Hungary": [19.0, 47.0],
        }

        return MapColoringPuzzle(europe_regions, coords, name="Europe Map")

    @staticmethod
    def from_adjacency(regions, coordinates=None, name="Custom Map"):
        """Create a puzzle from an adjacency dictionary and optional coordinates."""
        return MapColoringPuzzle(regions, coordinates, name=name)
