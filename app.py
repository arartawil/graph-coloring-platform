"""
Graph Coloring Platform - Main Streamlit Application
A comprehensive platform for exploring graph coloring puzzles and algorithms
"""

import streamlit as st

# ============================================================================
# Page Configuration - MUST BE FIRST STREAMLIT COMMAND
# ============================================================================
st.set_page_config(
    page_title="Graph Coloring Platform",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/graph-coloring-platform',
        'Report a bug': "https://github.com/yourusername/graph-coloring-platform/issues",
        'About': "# Graph Coloring Research Platform\nVersion 1.0.0"
    }
)

# Now import other modules
import sys
import os
import json
from pathlib import Path
import time
import traceback
from typing import Optional, Dict, Any, List

import networkx as nx
import pandas as pd
import plotly.express as px

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
IMPORTS_AVAILABLE = True
try:
    from database.database import DatabaseManager, initialize_database
    from algorithms import (
        greedy_coloring, dsatur_coloring, welsh_powell_coloring,
        tabu_search_coloring, dqn_coloring, DQN_AVAILABLE
    )
    from puzzles import (
        SudokuPuzzle, NQueensPuzzle, KakuroPuzzle, FutoshikiPuzzle,
        LatinSquarePuzzle, MapColoringPuzzle, CustomGraphPuzzle
    )
    from puzzles.futoshiki import generate_futoshiki, futoshiki_to_graph
    from puzzles.kakuro import generate_kakuro, kakuro_to_graph
    from puzzles.latin_square import latin_square_to_graph
    from utils.visualization import (
        visualize_graph_plotly, visualize_sudoku, plot_algorithm_comparison
    )
    from utils.graph_operations import (
        validate_coloring, count_colors, get_graph_stats, 
        compare_colorings
    )
except ImportError as e:
    IMPORTS_AVAILABLE = False
    st.error(f"Import Error: {e}")
    st.info("Make sure all required packages are installed: pip install -r requirements.txt")
    st.stop()

# ============================================================================
# Styling
# ============================================================================

def configure_page():
    """Configure custom page styling"""
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .success-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
        }
        .warning-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #007bff;
        }
        </style>
    """, unsafe_allow_html=True)


# ============================================================================
# Session State Management
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # User session
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    
    if 'username' not in st.session_state:
        st.session_state.username = "Guest"
    
    # Current selections
    if 'selected_puzzle' not in st.session_state:
        st.session_state.selected_puzzle = None
    
    if 'selected_puzzle_type' not in st.session_state:
        st.session_state.selected_puzzle_type = 'sudoku'
    
    if 'selected_algorithm' not in st.session_state:
        st.session_state.selected_algorithm = 'greedy'
    
    # Results cache
    if 'results_cache' not in st.session_state:
        st.session_state.results_cache = {}
    
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    # Graph data
    if 'current_graph' not in st.session_state:
        st.session_state.current_graph = None
    
    if 'current_coloring' not in st.session_state:
        st.session_state.current_coloring = None

    if 'current_positions' not in st.session_state:
        st.session_state.current_positions = None

    if 'generated_puzzle' not in st.session_state:
        st.session_state.generated_puzzle = None
    
    # Database connection
    if 'db' not in st.session_state:
        st.session_state.db = None
    
    # Page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Home'


def get_database() -> DatabaseManager:
    """Get or create database connection"""
    if st.session_state.db is None:
        try:
            db = DatabaseManager("graph_coloring.db")
            db.connect()
            # Initialize schema if needed
            if not Path("graph_coloring.db").exists():
                db.initialize_schema()
            st.session_state.db = db
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return None
    
    return st.session_state.db


# ============================================================================
# Navigation
# ============================================================================

def render_sidebar():
    """Render navigation sidebar"""
    with st.sidebar:
        st.title("🎨 Graph Coloring")
        st.markdown("---")
        
        # User info
        st.markdown(f"**User:** {st.session_state.username}")
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        pages = {
            "🏠 Home": "Home",
            "🧩 Puzzles": "Puzzles",
            "⚙️ Algorithms": "Algorithms",
            "📊 Results": "Results",
            "ℹ️ About": "About"
        }
        
        # Create navigation buttons
        for icon_name, page_name in pages.items():
            if st.button(icon_name, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
                st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.subheader("Quick Stats")
        db = get_database()
        if db:
            try:
                stats = db.get_database_statistics()
                st.metric("Total Puzzles", stats.get('puzzles_count', 0))
                st.metric("Results Stored", stats.get('results_count', 0))
                st.metric("Evaluations", stats.get('user_evaluations_count', 0))
            except Exception as e:
                st.caption(f"Stats unavailable: {e}")
        
        st.markdown("---")
        
        # Settings
        with st.expander("⚙️ Settings"):
            theme = st.selectbox("Theme", ["Light", "Dark", "Auto"])
            show_advanced = st.checkbox("Show Advanced Options", value=False)
            st.session_state.show_advanced = show_advanced


# ============================================================================
# Page: Home
# ============================================================================

def page_home():
    """Render home page"""
    st.title("🎨 Graph Coloring Research Platform")
    st.markdown("### Welcome to the comprehensive platform for exploring graph coloring puzzles and algorithms!")
    
    # Hero section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🧩 Multiple Puzzles</h3>
            <p>Explore Sudoku, N-Queens, Map Coloring, and custom graphs</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚙️ Advanced Algorithms</h3>
            <p>Compare greedy, DSatur, Welsh-Powell, Tabu Search, and DQN</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Analytics</h3>
            <p>Visualize results and compare algorithm performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Quick Start Guide")
        
        with st.expander("1️⃣ Choose a Puzzle", expanded=True):
            st.markdown("""
            Navigate to the **Puzzles** page to:
            - Select from pre-built puzzles (Sudoku, N-Queens, Map Coloring, etc.)
            - Create custom graph puzzles
            - Import puzzle data
            - View puzzle properties and constraints
            """)
        
        with st.expander("2️⃣ Select an Algorithm"):
            st.markdown("""
            Go to the **Algorithms** page to:
            - Choose from multiple coloring algorithms
            - Configure algorithm parameters
            - Run single or multiple algorithms
            - Compare algorithm performance
            
            **Available Algorithms:**
            - **Greedy**: Fast, simple approach
            - **DSatur**: Degree of saturation heuristic
            - **Welsh-Powell**: Sorts by degree first
            - **Tabu Search**: Metaheuristic optimization
            - **DQN**: Deep reinforcement learning
            """)
        
        with st.expander("3️⃣ Analyze Results"):
            st.markdown("""
            Visit the **Results** page to:
            - View coloring solutions
            - Analyze performance metrics
            - Compare algorithm efficiency
            - Export results for research
            - Generate visualizations
            """)
    
    with col2:
        st.subheader("📚 Resources")
        
        st.markdown("""
        **Documentation**
        - [User Guide](#)
        - [API Reference](#)
        - [Tutorials](#)
        
        **Research**
        - [Algorithm Comparison](#)
        - [Performance Analysis](#)
        - [Publications](#)
        
        **Support**
        - [FAQ](#)
        - [Report Issues](#)
        - [Community Forum](#)
        """)
        
        st.markdown("---")
        
        st.info("💡 **Tip:** Start with a simple puzzle like Map Coloring to understand the basics!")
    
    # Platform features
    st.markdown("---")
    st.subheader("✨ Platform Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🎯 Interactive Visualization**")
        st.caption("Real-time graph visualization with coloring results")
    
    with col2:
        st.markdown("**⚡ Fast Computation**")
        st.caption("Optimized algorithms for quick results")
    
    with col3:
        st.markdown("**💾 Data Persistence**")
        st.caption("Save puzzles, results, and evaluations")
    
    with col4:
        st.markdown("**📈 Research Tools**")
        st.caption("Export data and generate reports")
    
    # Recent activity
    st.markdown("---")
    st.subheader("📋 Recent Activity")
    
    db = get_database()
    if db:
        try:
            recent_results = db.get_all_results(limit=5)
            if recent_results:
                for result in recent_results:
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
                        with col1:
                            st.caption(f"Puzzle ID: {result['puzzle_id']}")
                        with col2:
                            st.caption(f"Algorithm: {result['algorithm_name']}")
                        with col3:
                            st.caption(f"Colors: {result['colors_used']}")
                        with col4:
                            st.caption(f"{result['execution_time']:.3f}s")
            else:
                st.info("No recent results. Start by solving a puzzle!")
        except Exception as e:
            st.warning(f"Could not load recent activity: {e}")


# ============================================================================
# Page: Puzzles
# ============================================================================

def page_puzzles():
    """Render puzzles page"""
    st.title("🧩 Puzzles")
    
    tab1, tab2, tab3 = st.tabs(["Browse Puzzles", "Create Puzzle", "My Puzzles"])
    
    with tab1:
        render_browse_puzzles()
    
    with tab2:
        render_create_puzzle()
    
    with tab3:
        render_my_puzzles()


def render_browse_puzzles():
    """Render puzzle browser"""
    st.subheader("Browse Available Puzzles")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        puzzle_type = st.selectbox(
            "Puzzle Type",
            ["All", "sudoku", "nqueens", "kakuro", "futoshiki", 
             "latin_square", "map_coloring", "custom_graph"]
        )
    
    with col2:
        difficulty = st.selectbox(
            "Difficulty",
            ["All", "easy", "medium", "hard", "expert"]
        )
    
    with col3:
        st.write("")  # Spacing
        st.write("")
        if st.button("🔍 Search", use_container_width=True):
            st.rerun()
    
    # Display puzzles
    db = get_database()
    if db:
        try:
            filters = {}
            if puzzle_type != "All":
                filters['puzzle_type'] = puzzle_type
            if difficulty != "All":
                filters['difficulty'] = difficulty
            
            puzzles = db.list_public_puzzles(
                puzzle_type=puzzle_type if puzzle_type != "All" else None,
                difficulty=difficulty if difficulty != "All" else None,
                limit=20
            )
            
            if puzzles:
                st.success(f"Found {len(puzzles)} puzzles")
                
                for puzzle in puzzles:
                    with st.expander(f"📊 {puzzle['name']} ({puzzle['puzzle_type']})"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Description:** {puzzle.get('description', 'No description')}")
                            st.write(f"**Vertices:** {puzzle['num_vertices']} | **Edges:** {puzzle['num_edges']}")
                            st.write(f"**Difficulty:** {puzzle.get('difficulty_level', 'N/A')}")
                            st.write(f"**Views:** {puzzle.get('views_count', 0)} | **Attempts:** {puzzle.get('attempts_count', 0)}")
                        
                        with col2:
                            if st.button("Load Puzzle", key=f"load_{puzzle['id']}"):
                                st.session_state.selected_puzzle = puzzle['id']
                                st.session_state.current_page = 'Algorithms'
                                st.success("Puzzle loaded! Navigate to Algorithms page.")
            else:
                st.info("No puzzles found. Try different filters or create your own!")
        
        except Exception as e:
            st.error(f"Error loading puzzles: {e}")


def render_create_puzzle():
    """Render puzzle creation interface"""
    st.subheader("Create New Puzzle")

    col_type, col_meta = st.columns([3, 2])
    with col_type:
        puzzle_type = st.selectbox(
            "Select Puzzle Type",
            ["map_coloring", "custom_graph", "sudoku", "nqueens", "latin_square", "futoshiki", "kakuro"],
            index=0
        )
    with col_meta:
        is_public = st.checkbox("Make puzzle public", value=False)

    st.session_state.selected_puzzle_type = puzzle_type

    default_name = f"{puzzle_type.replace('_', ' ').title()} puzzle"
    puzzle_name = st.text_input("Puzzle Name", value=default_name)
    description = st.text_area("Description", placeholder="Add a short description (optional)")
    difficulty = st.selectbox("Difficulty (optional)", ["None", "easy", "medium", "hard", "expert"], index=0)

    st.markdown("---")

    if puzzle_type == "sudoku":
        st.info("Sudoku puzzles are 9x9 grids with specific constraints")
        if st.button("Generate Sudoku Puzzle", type="primary"):
            try:
                puzzle = SudokuPuzzle()
                handle_generated_puzzle(
                    graph=puzzle.to_graph(),
                    puzzle_type="sudoku",
                    name=puzzle_name,
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=None,
                    metadata={"type": "sudoku"}
                )
            except Exception as e:
                st.error(f"Error creating Sudoku: {e}")

    elif puzzle_type == "nqueens":
        n = st.slider("Board Size (N)", min_value=4, max_value=20, value=8)
        if st.button("Generate N-Queens Puzzle", type="primary"):
            try:
                puzzle = NQueensPuzzle(n)
                handle_generated_puzzle(
                    graph=puzzle.to_graph(),
                    puzzle_type="nqueens",
                    name=puzzle_name or f"{n}-Queens",
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=None,
                    metadata={"board_size": n}
                )
            except Exception as e:
                st.error(f"Error creating N-Queens: {e}")

    elif puzzle_type == "map_coloring":
        map_mode = st.selectbox(
            "Generation mode",
            ["Random planar (Delaunay)", "USA (simplified)", "Europe (simplified)", "Custom adjacency"]
        )

        if map_mode == "Random planar (Delaunay)":
            num_regions = st.slider("Number of regions", min_value=6, max_value=40, value=12)
            seed = st.number_input("Random seed", value=42, step=1)
        elif map_mode == "Custom adjacency":
            custom_adj = st.text_area(
                "Adjacency as JSON (region -> neighbors)",
                value='{"RegionA": ["RegionB"], "RegionB": ["RegionA", "RegionC"]}'
            )
            custom_coords = st.text_area(
                "Optional coordinates as JSON (region -> [x, y])",
                value=''
            )

        if st.button("Generate Map Coloring Puzzle", type="primary"):
            try:
                if map_mode == "Random planar (Delaunay)":
                    puzzle = MapColoringPuzzle.create_random_planar(num_regions=num_regions, seed=int(seed))
                elif map_mode == "USA (simplified)":
                    puzzle = MapColoringPuzzle.create_usa_map()
                elif map_mode == "Europe (simplified)":
                    puzzle = MapColoringPuzzle.create_europe_map()
                else:
                    regions = json.loads(custom_adj) if custom_adj else {}
                    coords = json.loads(custom_coords) if custom_coords else None
                    puzzle = MapColoringPuzzle.from_adjacency(regions, coords, name="Custom Map")

                handle_generated_puzzle(
                    graph=puzzle.to_graph(),
                    puzzle_type="map_coloring",
                    name=puzzle_name or puzzle.name,
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=puzzle.positions,
                    metadata={"source": map_mode}
                )
            except Exception as e:
                st.error(f"Error creating map: {e}")

    elif puzzle_type == "custom_graph":
        st.info("Generate known graph families or random graphs.")
        graph_type = st.selectbox(
            "Graph Type",
            ["Random Erdos-Renyi", "Complete", "Cycle", "Wheel", "Bipartite", "Petersen"],
            index=0
        )

        if graph_type == "Random Erdos-Renyi":
            n = st.slider("Number of vertices", min_value=3, max_value=80, value=15)
            p = st.slider("Edge probability", min_value=0.05, max_value=1.0, value=0.25)
            seed = st.number_input("Random seed", value=42, step=1)
        elif graph_type == "Complete":
            n = st.slider("Number of vertices", min_value=2, max_value=50, value=6)
        elif graph_type == "Cycle":
            n = st.slider("Number of vertices", min_value=3, max_value=60, value=10)
        elif graph_type == "Wheel":
            n = st.slider("Number of vertices", min_value=4, max_value=40, value=8)
        elif graph_type == "Bipartite":
            n1 = st.slider("Left partition", min_value=2, max_value=40, value=5)
            n2 = st.slider("Right partition", min_value=2, max_value=40, value=5)
            p = st.slider("Cross-edge probability", min_value=0.05, max_value=1.0, value=0.3)
            seed = st.number_input("Random seed", value=21, step=1)

        if st.button("Generate Graph", type="primary"):
            try:
                if graph_type == "Random Erdos-Renyi":
                    puzzle = CustomGraphPuzzle.create_random_graph(n, p, seed=int(seed))
                    positions = puzzle.compute_layout(seed=int(seed))
                elif graph_type == "Complete":
                    puzzle = CustomGraphPuzzle.create_complete_graph(n)
                    positions = puzzle.compute_layout()
                elif graph_type == "Cycle":
                    puzzle = CustomGraphPuzzle.create_cycle_graph(n)
                    positions = puzzle.compute_layout(layout="circular")
                elif graph_type == "Wheel":
                    puzzle = CustomGraphPuzzle.create_wheel_graph(n)
                    positions = puzzle.compute_layout(layout="shell")
                elif graph_type == "Bipartite":
                    puzzle = CustomGraphPuzzle.create_bipartite_graph(n1, n2, p=p, seed=int(seed))
                    positions = puzzle.compute_layout(layout="kamada_kawai")
                else:
                    puzzle = CustomGraphPuzzle.create_petersen_graph()
                    positions = puzzle.compute_layout(layout="kamada_kawai")

                handle_generated_puzzle(
                    graph=puzzle.to_graph(),
                    puzzle_type="custom_graph",
                    name=puzzle_name or puzzle.name,
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=positions,
                    metadata={"graph_type": graph_type}
                )
            except Exception as e:
                st.error(f"Error creating custom graph: {e}")

    elif puzzle_type == "latin_square":
        size = st.slider("Size (n x n)", min_value=3, max_value=12, value=5)
        if st.button("Generate Latin Square Graph", type="primary"):
            try:
                graph = latin_square_to_graph(size)
                handle_generated_puzzle(
                    graph=graph,
                    puzzle_type="latin_square",
                    name=puzzle_name,
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=None,
                    metadata={"size": size}
                )
            except Exception as e:
                st.error(f"Error creating Latin square: {e}")

    elif puzzle_type == "futoshiki":
        size = st.slider("Grid size", min_value=3, max_value=9, value=5)
        if st.button("Generate Futoshiki Graph", type="primary"):
            try:
                puzzle = generate_futoshiki(size)
                graph = futoshiki_to_graph(puzzle)
                handle_generated_puzzle(
                    graph=graph,
                    puzzle_type="futoshiki",
                    name=puzzle_name,
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=None,
                    metadata={"size": size}
                )
            except Exception as e:
                st.error(f"Error creating Futoshiki: {e}")

    elif puzzle_type == "kakuro":
        rows = st.slider("Rows", min_value=5, max_value=15, value=8)
        cols = st.slider("Columns", min_value=5, max_value=15, value=8)
        if st.button("Generate Kakuro Graph", type="primary"):
            try:
                puzzle = generate_kakuro(rows, cols)
                graph = kakuro_to_graph(puzzle)
                handle_generated_puzzle(
                    graph=graph,
                    puzzle_type="kakuro",
                    name=puzzle_name,
                    description=description,
                    difficulty=difficulty,
                    is_public=is_public,
                    positions=None,
                    metadata={"rows": rows, "cols": cols}
                )
            except Exception as e:
                st.error(f"Error creating Kakuro: {e}")

    render_generated_puzzle_panel()


def render_my_puzzles():
    """Render user's puzzles"""
    st.subheader("My Puzzles")
    
    if st.session_state.user_id is None:
        st.warning("Please log in to view your puzzles")
        return
    
    db = get_database()
    if db:
        try:
            puzzles = db.get_user_puzzles(st.session_state.user_id)
            
            if puzzles:
                st.success(f"You have {len(puzzles)} puzzles")
                
                for puzzle in puzzles:
                    with st.expander(f"{puzzle['name']} - {puzzle['puzzle_type']}"):
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"Created: {puzzle['created_at']}")
                            st.write(f"Vertices: {puzzle['num_vertices']}, Edges: {puzzle['num_edges']}")
                        
                        with col2:
                            if st.button("Load", key=f"my_load_{puzzle['id']}"):
                                st.session_state.selected_puzzle = puzzle['id']
                                st.rerun()
                        
                        with col3:
                            if st.button("Delete", key=f"del_{puzzle['id']}"):
                                db.delete_puzzle(puzzle['id'])
                                st.success("Puzzle deleted!")
                                st.rerun()
            else:
                st.info("You haven't created any puzzles yet!")
        
        except Exception as e:
            st.error(f"Error loading puzzles: {e}")


def handle_generated_puzzle(graph, puzzle_type, name, description, difficulty, is_public, positions=None, metadata=None):
    """Persist generated puzzle in session for preview and saving."""
    st.session_state.current_graph = graph
    st.session_state.current_coloring = None
    st.session_state.current_positions = positions
    st.session_state.generated_puzzle = {
        "graph": graph,
        "positions": positions,
        "puzzle_type": puzzle_type,
        "name": name or f"{puzzle_type} puzzle",
        "description": description or "",
        "difficulty": difficulty,
        "is_public": is_public,
        "metadata": metadata or {},
    }
    st.success("✅ Puzzle generated. Preview and save below.")


def serialize_graph(graph, positions=None):
    """Serialize a NetworkX graph for storage or download."""
    payload = {"nodes": [], "edges": [], "coordinates": {}}
    pos_lookup = positions or {node: data.get("pos") for node, data in graph.nodes(data=True)}

    for node, data in graph.nodes(data=True):
        payload["nodes"].append(str(node))
        pos = pos_lookup.get(node)
        if pos is not None:
            payload["coordinates"][str(node)] = [float(pos[0]), float(pos[1])]

    for u, v in graph.edges():
        payload["edges"].append([str(u), str(v)])

    return payload


def render_generated_puzzle_panel():
    generated = st.session_state.get("generated_puzzle")
    if not generated:
        return

    graph = generated["graph"]
    positions = generated.get("positions") or st.session_state.get("current_positions")

    st.markdown("---")
    st.subheader("Preview & Save")

    stats = get_graph_stats(graph)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vertices", stats.get("num_nodes", 0))
    col2.metric("Edges", stats.get("num_edges", 0))
    col3.metric("Density", f"{stats.get('density', 0):.3f}")
    col4.metric("Connected", "Yes" if stats.get("is_connected") else "No")

    try:
        fig = visualize_graph_plotly(graph, title=generated.get("name", "Graph"), positions=positions)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Visualization error: {e}")

    graph_json = json.dumps(serialize_graph(graph, positions), indent=2)
    st.download_button(
        label="Download graph JSON",
        data=graph_json,
        file_name=f"{generated.get('puzzle_type', 'puzzle')}.json",
        mime="application/json"
    )

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("💾 Save to database", type="primary"):
            save_generated_puzzle(generated, stats)
    with action_col2:
        if st.button("➡️ Use in Algorithms"):
            st.session_state.current_page = "Algorithms"
            st.rerun()


def save_generated_puzzle(generated, stats):
    db = get_database()
    if not db:
        st.error("Database not available")
        return

    graph_data = serialize_graph(generated["graph"], generated.get("positions"))
    difficulty = generated.get("difficulty")
    difficulty = difficulty if difficulty and difficulty != "None" else None

    try:
        puzzle_id = db.create_puzzle(
            user_id=st.session_state.user_id or 1,
            name=generated.get("name", "Untitled"),
            puzzle_type=generated.get("puzzle_type", "custom_graph"),
            num_vertices=stats.get("num_nodes", 0),
            num_edges=stats.get("num_edges", 0),
            graph_data=graph_data,
            description=generated.get("description"),
            difficulty_level=difficulty,
            density=stats.get("density"),
            chromatic_number=None,
            initial_constraints=generated.get("metadata"),
            is_public=generated.get("is_public", False)
        )
        st.success(f"Puzzle saved with ID {puzzle_id}")
    except Exception as e:
        st.error(f"Failed to save puzzle: {e}")


# ============================================================================
# Page: Algorithms
# ============================================================================


def _deserialize_graph(data: Dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    if not data:
        return graph
    graph.add_nodes_from(data.get("nodes", []))
    graph.add_edges_from([tuple(edge) for edge in data.get("edges", [])])
    coords = data.get("coordinates", {})
    if coords:
        nx.set_node_attributes(graph, {k: tuple(v) for k, v in coords.items()}, "pos")
    if data.get("chromatic_number"):
        graph.graph["chromatic_number"] = data["chromatic_number"]
    return graph


def _load_puzzle_graph(puzzle: Dict[str, Any]) -> nx.Graph:
    raw = puzzle.get("graph_data")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return _deserialize_graph(raw)


@st.cache_data(show_spinner=False)
def _run_algorithm_cached(algo_name: str, nodes: tuple, edges: tuple, chromatic: Optional[int] = None):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    if chromatic is not None:
        g.graph["chromatic_number"] = chromatic

    algo_map = {
        "greedy": greedy_coloring,
        "dsatur": dsatur_coloring,
        "welsh_powell": welsh_powell_coloring,
        "tabu_search": tabu_search_coloring,
    }
    func = algo_map[algo_name]

    start = time.perf_counter()
    coloring = func(g)
    exec_time = time.perf_counter() - start
    valid = validate_coloring(g, coloring)
    colors_used = count_colors(coloring)
    chromatic_val = g.graph.get("chromatic_number")
    optimal = chromatic_val is not None and colors_used == chromatic_val

    return {
        "algorithm": algo_name,
        "coloring": coloring,
        "colors_used": colors_used,
        "execution_time": exec_time,
        "is_valid": valid,
        "optimal": optimal,
        "chromatic_number": chromatic_val,
    }


def page_algorithms():
    """Render algorithms page with comparison tools."""
    st.title("⚙️ Algorithms")

    db = get_database()
    user_puzzles = []
    if db and st.session_state.user_id:
        try:
            user_puzzles = db.get_user_puzzles(st.session_state.user_id)
        except Exception:
            user_puzzles = []

    with st.expander("Select puzzle", expanded=True):
        options = ["Current session"] + [f"{p['id']}: {p['name']}" for p in user_puzzles]
        choice = st.selectbox("Puzzle", options)
        if choice != "Current session":
            selected_id = int(choice.split(":")[0])
            puzzle = next((p for p in user_puzzles if p["id"] == selected_id), None)
            if puzzle:
                loaded_graph = _load_puzzle_graph(puzzle)
                st.session_state.current_graph = loaded_graph
                st.session_state.current_positions = nx.get_node_attributes(loaded_graph, "pos") or None
                st.session_state.selected_puzzle = selected_id
                chrom = puzzle.get("chromatic_number")
                if chrom:
                    loaded_graph.graph["chromatic_number"] = chrom
                st.info(f"Loaded puzzle {puzzle['name']} (vertices: {loaded_graph.number_of_nodes()})")

    if st.session_state.current_graph is None:
        st.warning("⚠️ No puzzle selected! Please create or load a puzzle first.")
        return

    graph = st.session_state.current_graph
    stats = get_graph_stats(graph)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vertices", stats.get("num_nodes", 0))
    c2.metric("Edges", stats.get("num_edges", 0))
    c3.metric("Density", f"{stats.get('density', 0):.3f}")
    c4.metric("Avg Degree", f"{stats.get('avg_degree', 0):.2f}")

    st.markdown("---")

    with st.expander("Graph preview", expanded=False):
        try:
            preview_fig = visualize_graph_plotly(graph, positions=st.session_state.current_positions, title="Selected puzzle")
            st.plotly_chart(preview_fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Preview unavailable: {e}")

    st.subheader("Select algorithms")
    select_all = st.checkbox("Select all", value=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        use_greedy = st.checkbox("Greedy", value=select_all)
        use_dsatur = st.checkbox("DSatur", value=select_all)
    with col_b:
        use_wp = st.checkbox("Welsh-Powell", value=select_all)
        use_tabu = st.checkbox("Tabu Search", value=select_all)
    with col_c:
        if DQN_AVAILABLE:
            use_dqn = st.checkbox("DQN (experimental)", value=False)
        else:
            st.checkbox("DQN (requires torch)", value=False, disabled=True)
            st.caption("Install torch to enable DQN.")
            use_dqn = False

    if select_all:
        use_greedy = use_dsatur = use_wp = use_tabu = True

    selected_algos = [
        name for name, flag in [
            ("greedy", use_greedy),
            ("dsatur", use_dsatur),
            ("welsh_powell", use_wp),
            ("tabu_search", use_tabu),
            ("dqn", use_dqn),
        ] if flag
    ]

    st.subheader("Run comparison")
    run_col, save_col = st.columns([2, 1])
    with run_col:
        run_btn = st.button("🔄 Run selected algorithms", type="primary")
    with save_col:
        save_to_db = st.checkbox("Save results to DB", value=False)

    results = []
    if run_btn and selected_algos:
        nodes = tuple(graph.nodes())
        edges = tuple((u, v) for u, v in graph.edges())
        chromatic_number = graph.graph.get("chromatic_number")
        progress = st.progress(0)
        for idx, algo in enumerate(selected_algos):
            if algo == "dqn":
                if not DQN_AVAILABLE:
                    st.warning("DQN is not available. Install torch to enable.")
                    continue
                start = time.perf_counter()
                coloring = dqn_coloring(graph)
                exec_time = time.perf_counter() - start
                valid = validate_coloring(graph, coloring)
                res = {
                    "algorithm": "dqn",
                    "coloring": coloring,
                    "colors_used": count_colors(coloring),
                    "execution_time": exec_time,
                    "is_valid": valid,
                    "optimal": None,
                    "chromatic_number": graph.graph.get("chromatic_number"),
                }
            else:
                res = _run_algorithm_cached(algo, nodes, edges, chromatic_number)
            results.append(res)
            progress.progress((idx + 1) / len(selected_algos))

        st.session_state.last_result = results

        if save_to_db and db and st.session_state.selected_puzzle:
            for res in results:
                try:
                    db.save_result(
                        puzzle_id=st.session_state.selected_puzzle,
                        user_id=st.session_state.user_id,
                        algorithm_name=res["algorithm"],
                        colors_used=res["colors_used"],
                        execution_time=res["execution_time"],
                        coloring_result=res["coloring"],
                        is_optimal=res.get("optimal", False),
                        is_valid=res.get("is_valid", True),
                    )
                except Exception:
                    pass

    if not results and st.session_state.get("last_result"):
        results = st.session_state.last_result

    if results:
        st.markdown("---")
        st.subheader("📊 Results")
        df = pd.DataFrame([
            {
                "algorithm": r["algorithm"],
                "colors_used": r["colors_used"],
                "execution_time": r["execution_time"],
                "optimal": r.get("optimal"),
                "valid": r.get("is_valid"),
            }
            for r in results
        ])

        best_idx = df["colors_used"].idxmin()
        df_sorted = df.sort_values(["colors_used", "execution_time"])
        st.dataframe(df_sorted, use_container_width=True)

        best_row = df.loc[best_idx]
        b1, b2 = st.columns(2)
        b1.metric("Fewest colors", f"{best_row['algorithm']} ({best_row['colors_used']})")
        fastest = df.loc[df["execution_time"].idxmin()]
        b2.metric("Fastest", f"{fastest['algorithm']} ({fastest['execution_time']:.4f}s)")

        # Best solution highlight
        best_coloring = results[best_idx].get("coloring", {}) if isinstance(best_idx, int) else results[0].get("coloring", {})
        st.markdown("### Best solution")
        best_fig = None
        try:
            best_fig = visualize_graph_plotly(graph, best_coloring, title=f"Best: {best_row['algorithm'].upper()}", positions=st.session_state.current_positions)
            st.plotly_chart(best_fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Best visualization error: {e}")

        # Visualizations
        st.subheader("Visualizations")
        viz_cols = st.columns(len(results))
        for idx, res in enumerate(results):
            with viz_cols[idx]:
                try:
                    fig = visualize_graph_plotly(graph, res["coloring"], title=res["algorithm"].upper(), positions=st.session_state.current_positions)
                    st.plotly_chart(fig, use_container_width=True)
                    try:
                        img = fig.to_image(format="png")
                        st.download_button(
                            f"⬇️ {res['algorithm']} image",
                            data=img,
                            file_name=f"{res['algorithm']}_coloring.png",
                            mime="image/png",
                            key=f"download_{res['algorithm']}_{idx}"
                        )
                    except Exception:
                        st.caption("Install kaleido to export images.")
                except Exception as e:
                    st.caption(f"Viz error: {e}")

        scatter_fig = px.scatter(
            df,
            x="execution_time",
            y="colors_used",
            color="algorithm",
            title="Performance (time vs colors)",
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

        # Color distribution for best solution
        if best_coloring:
            counts = pd.Series(best_coloring).value_counts().sort_index()
            hist_fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Color", "y": "Frequency"}, title="Color distribution (best)")
            st.plotly_chart(hist_fig, use_container_width=True)

        # Export options
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv_data, file_name="algorithm_comparison.csv", mime="text/csv")

        try:
            image_bytes = scatter_fig.to_image(format="png")
            st.download_button("⬇️ Download performance chart", data=image_bytes, file_name="performance.png", mime="image/png")
        except Exception:
            st.caption("Install kaleido to enable image exports.")

        if best_fig:
            try:
                best_image = best_fig.to_image(format="png")
                st.download_button("⬇️ Download best coloring image", data=best_image, file_name="best_coloring.png", mime="image/png")
            except Exception:
                st.caption("Install kaleido to export coloring images.")


# ============================================================================
# Page: Results
# ============================================================================

def page_results():
    """Render results page"""
    st.title("📊 Results & Analytics")
    
    db = get_database()
    if not db:
        st.error("Database not available")
        return
    
    tab1, tab2 = st.tabs(["View Results", "Analytics"])
    
    with tab1:
        render_results_viewer(db)
    
    with tab2:
        render_analytics(db)


def render_results_viewer(db: DatabaseManager):
    """Render results viewer"""
    st.subheader("Stored Results")
    
    try:
        results = db.get_all_results(limit=50)
        
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                algo_filter = st.multiselect("Filter by Algorithm", 
                                            df['algorithm_name'].unique())
            with col2:
                puzzle_filter = st.multiselect("Filter by Puzzle ID", 
                                              df['puzzle_id'].unique())
            
            # Apply filters
            filtered_df = df.copy()
            if algo_filter:
                filtered_df = filtered_df[filtered_df['algorithm_name'].isin(algo_filter)]
            if puzzle_filter:
                filtered_df = filtered_df[filtered_df['puzzle_id'].isin(puzzle_filter)]
            
            st.dataframe(filtered_df[['id', 'puzzle_id', 'algorithm_name', 
                                     'colors_used', 'execution_time', 'timestamp']], 
                        use_container_width=True)
        else:
            st.info("No results yet. Run some algorithms to see results here!")
    
    except Exception as e:
        st.error(f"Error loading results: {e}")


def render_analytics(db: DatabaseManager):
    """Render analytics dashboard"""
    st.subheader("Performance Analytics")
    
    try:
        summary = db.get_algorithm_performance_summary()
        
        if summary:
            import pandas as pd
            df = pd.DataFrame(summary)
            
            st.markdown("### Algorithm Performance Summary")
            st.dataframe(df, use_container_width=True)
            
            # Statistics
            stats = db.get_evaluation_statistics()
            if stats:
                st.markdown("### Evaluation Statistics")
                col1, col2, col3 = st.columns(3)
                
                col1.metric("Total Evaluations", stats.get('total_evaluations', 0))
                col2.metric("Avg Improvement", f"{stats.get('avg_improvement', 0):.1f}%")
                col3.metric("Avg Usability", f"{stats.get('avg_usability', 0):.2f}/5")
        else:
            st.info("No analytics data available yet")
    
    except Exception as e:
        st.error(f"Error loading analytics: {e}")


# ============================================================================
# Page: About
# ============================================================================

def page_about():
    """Render about page"""
    st.title("ℹ️ About")
    
    st.markdown("""
    ## Graph Coloring Research Platform
    
    **Version:** 1.0.0  
    **Released:** December 2025
    
    ### Overview
    
    This platform provides a comprehensive environment for exploring graph coloring problems,
    algorithms, and applications. It is designed for students, researchers, and enthusiasts
    interested in graph theory and combinatorial optimization.
    
    ### Features
    
    - 🧩 **Multiple Puzzle Types**: Sudoku, N-Queens, Map Coloring, and custom graphs
    - ⚙️ **Advanced Algorithms**: Greedy, DSatur, Welsh-Powell, Tabu Search, and DQN
    - 📊 **Visualization**: Interactive graph visualization with Plotly
    - 💾 **Data Persistence**: SQLite database for storing puzzles and results
    - 📈 **Analytics**: Performance comparison and evaluation tools
    
    ### Algorithms
    
    #### Greedy Coloring
    Simple sequential algorithm that assigns the first available color to each vertex.
    
    #### DSatur (Degree of Saturation)
    Heuristic that prioritizes vertices with the highest saturation degree.
    
    #### Welsh-Powell
    Pre-sorts vertices by degree before applying greedy coloring.
    
    #### Tabu Search
    Metaheuristic approach that explores the solution space systematically.
    
    #### DQN (Deep Q-Network)
    Reinforcement learning approach for graph coloring (experimental).
    
    ### Technology Stack
    
    - **Frontend**: Streamlit
    - **Backend**: Python, SQLite
    - **Visualization**: Plotly, Matplotlib
    - **Graph Operations**: NetworkX
    - **ML/RL**: PyTorch, Stable-Baselines3
    
    ### License
    
    MIT License - See LICENSE file for details
    
    ### Contact
    
    For questions, suggestions, or bug reports, please visit our GitHub repository
    or contact the development team.
    
    ---
    
    **Developed with ❤️ for the graph theory community**
    """)


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point"""
    try:
        # Configure custom styling (page_config already called at top)
        configure_page()
        
        # Initialize session state
        initialize_session_state()
        
        # Render sidebar
        render_sidebar()
        
        # Route to appropriate page
        page = st.session_state.current_page
        
        if page == "Home":
            page_home()
        elif page == "Puzzles":
            page_puzzles()
        elif page == "Algorithms":
            page_algorithms()
        elif page == "Results":
            page_results()
        elif page == "About":
            page_about()
        else:
            page_home()
    
    except Exception as e:
        st.error(f"Application Error: {e}")
        st.code(traceback.format_exc())
        st.info("Please refresh the page or report this issue if it persists.")


if __name__ == "__main__":
    main()
