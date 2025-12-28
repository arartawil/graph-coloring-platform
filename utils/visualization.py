# Visualization Utilities
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

def visualize_graph(graph, coloring=None, title="Graph", layout='spring'):
    """
    Visualize a graph with optional coloring using matplotlib.
    
    Args:
        graph: NetworkX graph
        coloring: Dictionary mapping nodes to colors
        title: Plot title
        layout: Layout algorithm ('spring', 'circular', 'random')
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'circular':
        pos = nx.circular_layout(graph)
    else:
        pos = nx.random_layout(graph)
    
    # Draw graph
    if coloring:
        node_colors = [coloring.get(node, 0) for node in graph.nodes()]
        nx.draw(graph, pos, node_color=node_colors, with_labels=True, 
                node_size=500, cmap=plt.cm.rainbow, ax=ax)
    else:
        nx.draw(graph, pos, with_labels=True, node_size=500, ax=ax)
    
    ax.set_title(title)
    return fig

def visualize_graph_plotly(graph, coloring=None, title="Graph", positions=None):
    """
    Create interactive graph visualization using Plotly.
    
    Args:
        graph: NetworkX graph
        coloring: Dictionary mapping nodes to colors
        title: Plot title
        positions: Optional dict of node -> (x, y) coordinates
        
    Returns:
        plotly figure
    """
    pos = positions or nx.get_node_attributes(graph, "pos")
    if not pos:
        pos = nx.spring_layout(graph)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        if coloring:
            node_colors.append(coloring.get(node, 0))
            node_text.append(f"Node: {node}<br>Color: {coloring.get(node, 0)}")
        else:
            node_colors.append(0)
            node_text.append(f"Node: {node}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[str(node) for node in graph.nodes()],
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Rainbow',
            color=node_colors,
            size=20,
            colorbar=dict(
                thickness=15,
                title='Color',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    
    return fig

def visualize_sudoku(grid):
    """
    Visualize a Sudoku grid.
    
    Args:
        grid: 9x9 numpy array
        
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid
    for i in range(10):
        linewidth = 3 if i % 3 == 0 else 1
        ax.plot([0, 9], [i, i], 'k-', linewidth=linewidth)
        ax.plot([i, i], [0, 9], 'k-', linewidth=linewidth)
    
    # Fill in numbers
    for i in range(9):
        for j in range(9):
            if grid[i, j] != 0:
                ax.text(j + 0.5, 8.5 - i, str(grid[i, j]), 
                       ha='center', va='center', fontsize=20)
    
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Sudoku Grid')
    
    return fig

def plot_algorithm_comparison(results_df):
    """
    Plot comparison of different algorithms.
    
    Args:
        results_df: Pandas DataFrame with columns ['algorithm', 'colors_used', 'time']
        
    Returns:
        matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors used
    ax1.bar(results_df['algorithm'], results_df['colors_used'])
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Colors Used')
    ax1.set_title('Number of Colors Used by Algorithm')
    ax1.tick_params(axis='x', rotation=45)
    
    # Execution time
    ax2.bar(results_df['algorithm'], results_df['time'])
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Execution Time by Algorithm')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig
