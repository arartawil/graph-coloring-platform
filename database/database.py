"""
SQLite Database Operations for Graph Coloring Platform
Provides comprehensive database management with CRUD operations
"""

import sqlite3
import os
import json
import logging
from typing import Optional, Dict, List, Any, Tuple, Union
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass


class DatabaseManager:
    """
    Comprehensive database manager for graph coloring platform.
    Supports context manager protocol for safe resource management.
    """
    
    def __init__(self, db_path: str = "graph_coloring.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    # =========================================================================
    # Connection Management
    # =========================================================================
    
    def connect(self) -> sqlite3.Connection:
        """
        Establish database connection with row factory.
        
        Returns:
            sqlite3.Connection object
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name
            self.conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
            logger.info(f"Connected to database: {self.db_path}")
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Failed to connect to database: {e}")
    
    def close(self):
        """Close database connection safely"""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Database connection closed")
            except sqlite3.Error as e:
                logger.error(f"Error closing database: {e}")
            finally:
                self.conn = None
    
    def initialize_schema(self, schema_file: str = "database/schema.sql") -> bool:
        """
        Initialize database schema from SQL file.
        
        Args:
            schema_file: Path to schema SQL file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.conn:
            raise DatabaseError("Database not connected")
        
        schema_path = Path(schema_file)
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_file}")
            return False
        
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = f.read()
            self.conn.executescript(schema)
            self.conn.commit()
            logger.info(f"Schema initialized from {schema_file}")
            return True
        except (sqlite3.Error, IOError) as e:
            logger.error(f"Schema initialization error: {e}")
            raise DatabaseError(f"Failed to initialize schema: {e}")
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions with automatic rollback.
        
        Usage:
            with db.transaction():
                db.execute_update(...)
        """
        if not self.conn:
            raise DatabaseError("Database not connected")
        
        try:
            yield self.conn
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    # =========================================================================
    # Generic Query Methods
    # =========================================================================
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Union[Tuple, Dict]] = None
    ) -> List[sqlite3.Row]:
        """
        Execute a SELECT query and return results.
        
        Args:
            query: SQL query string
            params: Query parameters (tuple or dict)
            
        Returns:
            List of Row objects
        """
        if not self.conn:
            raise DatabaseError("Database not connected")
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}\nQuery: {query}")
            raise DatabaseError(f"Query failed: {e}")
    
    def execute_update(
        self, 
        query: str, 
        params: Optional[Union[Tuple, Dict]] = None
    ) -> int:
        """
        Execute an INSERT/UPDATE/DELETE query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Last inserted row ID or affected rows count
        """
        if not self.conn:
            raise DatabaseError("Database not connected")
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.conn.commit()
            return cursor.lastrowid if cursor.lastrowid else cursor.rowcount
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Update execution error: {e}\nQuery: {query}")
            raise DatabaseError(f"Update failed: {e}")
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """
        Execute multiple queries with different parameters.
        
        Args:
            query: SQL query string
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        if not self.conn:
            raise DatabaseError("Database not connected")
        
        try:
            cursor = self.conn.cursor()
            cursor.executemany(query, params_list)
            self.conn.commit()
            return cursor.rowcount
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Batch execution error: {e}")
            raise DatabaseError(f"Batch update failed: {e}")
    
    # =========================================================================
    # Puzzle CRUD Operations
    # =========================================================================
    
    def create_puzzle(
        self,
        user_id: int,
        name: str,
        puzzle_type: str,
        num_vertices: int,
        num_edges: int,
        graph_data: Dict[str, Any],
        description: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        density: Optional[float] = None,
        chromatic_number: Optional[int] = None,
        initial_constraints: Optional[Dict] = None,
        is_public: bool = False
    ) -> int:
        """
        Create a new puzzle in the database.
        
        Args:
            user_id: ID of the user creating the puzzle
            name: Puzzle name
            puzzle_type: Type of puzzle (sudoku, nqueens, etc.)
            num_vertices: Number of vertices in the graph
            num_edges: Number of edges in the graph
            graph_data: Graph structure as dictionary
            description: Optional puzzle description
            difficulty_level: Optional difficulty (easy, medium, hard, expert)
            density: Optional graph density (0.0 to 1.0)
            chromatic_number: Optional chromatic number
            initial_constraints: Optional constraints as dictionary
            is_public: Whether puzzle is public
            
        Returns:
            ID of the created puzzle
        """
        query = """
        INSERT INTO puzzles (
            user_id, name, description, puzzle_type, difficulty_level,
            num_vertices, num_edges, density, chromatic_number,
            graph_data, initial_constraints, is_public
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            user_id,
            name,
            description,
            puzzle_type,
            difficulty_level,
            num_vertices,
            num_edges,
            density,
            chromatic_number,
            json.dumps(graph_data),
            json.dumps(initial_constraints) if initial_constraints else None,
            int(is_public)
        )
        
        puzzle_id = self.execute_update(query, params)
        logger.info(f"Created puzzle {puzzle_id}: {name}")
        return puzzle_id
    
    def get_puzzle(self, puzzle_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a puzzle by ID.
        
        Args:
            puzzle_id: Puzzle ID
            
        Returns:
            Dictionary with puzzle data or None if not found
        """
        query = "SELECT * FROM puzzles WHERE id = ?"
        results = self.execute_query(query, (puzzle_id,))
        
        if not results:
            return None
        
        row = results[0]
        puzzle = dict(row)
        
        # Parse JSON fields
        if puzzle.get('graph_data'):
            puzzle['graph_data'] = json.loads(puzzle['graph_data'])
        if puzzle.get('initial_constraints'):
            puzzle['initial_constraints'] = json.loads(puzzle['initial_constraints'])
        
        return puzzle
    
    def get_user_puzzles(
        self, 
        user_id: int, 
        include_private: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all puzzles created by a user.
        
        Args:
            user_id: User ID
            include_private: Whether to include private puzzles
            
        Returns:
            List of puzzle dictionaries
        """
        if include_private:
            query = "SELECT * FROM puzzles WHERE user_id = ? ORDER BY created_at DESC"
            params = (user_id,)
        else:
            query = "SELECT * FROM puzzles WHERE user_id = ? AND is_public = 1 ORDER BY created_at DESC"
            params = (user_id,)
        
        results = self.execute_query(query, params)
        
        puzzles = []
        for row in results:
            puzzle = dict(row)
            if puzzle.get('graph_data'):
                puzzle['graph_data'] = json.loads(puzzle['graph_data'])
            if puzzle.get('initial_constraints'):
                puzzle['initial_constraints'] = json.loads(puzzle['initial_constraints'])
            puzzles.append(puzzle)
        
        return puzzles
    
    def list_public_puzzles(
        self, 
        puzzle_type: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List public puzzles with optional filters.
        
        Args:
            puzzle_type: Optional filter by puzzle type
            difficulty: Optional filter by difficulty level
            limit: Maximum number of results
            offset: Result offset for pagination
            
        Returns:
            List of puzzle dictionaries
        """
        query = "SELECT * FROM puzzles WHERE is_public = 1"
        params = []
        
        if puzzle_type:
            query += " AND puzzle_type = ?"
            params.append(puzzle_type)
        
        if difficulty:
            query += " AND difficulty_level = ?"
            params.append(difficulty)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        results = self.execute_query(query, tuple(params))
        
        puzzles = []
        for row in results:
            puzzle = dict(row)
            # Don't include full graph_data for list views (performance)
            puzzle['graph_data'] = None
            puzzles.append(puzzle)
        
        return puzzles
    
    def update_puzzle(
        self, 
        puzzle_id: int, 
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update puzzle fields.
        
        Args:
            puzzle_id: Puzzle ID
            updates: Dictionary of fields to update
            
        Returns:
            True if successful, False otherwise
        """
        if not updates:
            return False
        
        # Build dynamic UPDATE query
        allowed_fields = {
            'name', 'description', 'difficulty_level', 'is_public',
            'is_verified', 'chromatic_number', 'graph_data', 'initial_constraints'
        }
        
        update_fields = []
        params = []
        
        for field, value in updates.items():
            if field in allowed_fields:
                update_fields.append(f"{field} = ?")
                # Convert dict/list to JSON for specific fields
                if field in ('graph_data', 'initial_constraints') and isinstance(value, (dict, list)):
                    value = json.dumps(value)
                params.append(value)
        
        if not update_fields:
            return False
        
        params.append(puzzle_id)
        query = f"UPDATE puzzles SET {', '.join(update_fields)} WHERE id = ?"
        
        rows_affected = self.execute_update(query, tuple(params))
        return rows_affected > 0
    
    def delete_puzzle(self, puzzle_id: int) -> bool:
        """
        Delete a puzzle (cascade deletes results).
        
        Args:
            puzzle_id: Puzzle ID
            
        Returns:
            True if deleted, False otherwise
        """
        query = "DELETE FROM puzzles WHERE id = ?"
        rows_affected = self.execute_update(query, (puzzle_id,))
        
        if rows_affected > 0:
            logger.info(f"Deleted puzzle {puzzle_id}")
            return True
        return False
    
    def increment_puzzle_views(self, puzzle_id: int) -> bool:
        """
        Increment puzzle views counter.
        
        Args:
            puzzle_id: Puzzle ID
            
        Returns:
            True if successful
        """
        query = "UPDATE puzzles SET views_count = views_count + 1 WHERE id = ?"
        return self.execute_update(query, (puzzle_id,)) > 0
    
    # =========================================================================
    # Results CRUD Operations
    # =========================================================================
    
    def save_result(
        self,
        puzzle_id: int,
        algorithm_name: str,
        colors_used: int,
        execution_time: float,
        coloring_result: Dict[str, int],
        user_id: Optional[int] = None,
        memory_usage: Optional[float] = None,
        iterations: Optional[int] = None,
        is_optimal: bool = False,
        is_valid: bool = True,
        algorithm_params: Optional[Dict] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Save algorithm execution result.
        
        Args:
            puzzle_id: Puzzle ID
            algorithm_name: Name of the algorithm used
            colors_used: Number of colors used in the solution
            execution_time: Time taken in seconds
            coloring_result: Node-to-color mapping
            user_id: Optional user ID
            memory_usage: Optional memory usage in MB
            iterations: Optional iteration count
            is_optimal: Whether the solution is optimal
            is_valid: Whether the solution is valid
            algorithm_params: Optional algorithm parameters
            notes: Optional notes
            
        Returns:
            ID of the created result
        """
        query = """
        INSERT INTO results (
            puzzle_id, user_id, algorithm_name, colors_used, execution_time,
            memory_usage, iterations, is_optimal, is_valid,
            coloring_result, algorithm_params, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            puzzle_id,
            user_id,
            algorithm_name,
            colors_used,
            execution_time,
            memory_usage,
            iterations,
            int(is_optimal),
            int(is_valid),
            json.dumps(coloring_result),
            json.dumps(algorithm_params) if algorithm_params else None,
            notes
        )
        
        result_id = self.execute_update(query, params)
        logger.info(f"Saved result {result_id} for puzzle {puzzle_id} using {algorithm_name}")
        return result_id
    
    def get_puzzle_results(
        self, 
        puzzle_id: int,
        algorithm_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all results for a puzzle.
        
        Args:
            puzzle_id: Puzzle ID
            algorithm_name: Optional filter by algorithm
            
        Returns:
            List of result dictionaries
        """
        if algorithm_name:
            query = """
            SELECT * FROM results 
            WHERE puzzle_id = ? AND algorithm_name = ?
            ORDER BY timestamp DESC
            """
            params = (puzzle_id, algorithm_name)
        else:
            query = "SELECT * FROM results WHERE puzzle_id = ? ORDER BY timestamp DESC"
            params = (puzzle_id,)
        
        results = self.execute_query(query, params)
        
        result_list = []
        for row in results:
            result = dict(row)
            if result.get('coloring_result'):
                result['coloring_result'] = json.loads(result['coloring_result'])
            if result.get('algorithm_params'):
                result['algorithm_params'] = json.loads(result['algorithm_params'])
            result_list.append(result)
        
        return result_list
    
    def get_algorithm_comparison(self, puzzle_id: int) -> Dict[str, Dict[str, Any]]:
        """
        Get comparison of all algorithms for a puzzle.
        
        Args:
            puzzle_id: Puzzle ID
            
        Returns:
            Dictionary mapping algorithm names to their best results
        """
        query = """
        SELECT 
            algorithm_name,
            MIN(colors_used) as best_colors,
            AVG(colors_used) as avg_colors,
            MIN(execution_time) as best_time,
            AVG(execution_time) as avg_time,
            COUNT(*) as runs,
            SUM(CASE WHEN is_optimal = 1 THEN 1 ELSE 0 END) as optimal_count
        FROM results
        WHERE puzzle_id = ?
        GROUP BY algorithm_name
        ORDER BY best_colors ASC, best_time ASC
        """
        
        results = self.execute_query(query, (puzzle_id,))
        
        comparison = {}
        for row in results:
            algo_name = row['algorithm_name']
            comparison[algo_name] = {
                'best_colors': row['best_colors'],
                'avg_colors': round(row['avg_colors'], 2),
                'best_time': round(row['best_time'], 4),
                'avg_time': round(row['avg_time'], 4),
                'total_runs': row['runs'],
                'optimal_solutions': row['optimal_count']
            }
        
        return comparison
    
    def get_all_results(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get all results with optional filters.
        
        Args:
            filters: Dictionary of filter conditions
                - puzzle_id: Filter by puzzle
                - algorithm_name: Filter by algorithm
                - min_colors: Minimum colors used
                - max_colors: Maximum colors used
                - is_optimal: Filter optimal solutions
            limit: Maximum results to return
            offset: Result offset
            
        Returns:
            List of result dictionaries
        """
        query = "SELECT * FROM results WHERE 1=1"
        params = []
        
        if filters:
            if 'puzzle_id' in filters:
                query += " AND puzzle_id = ?"
                params.append(filters['puzzle_id'])
            
            if 'algorithm_name' in filters:
                query += " AND algorithm_name = ?"
                params.append(filters['algorithm_name'])
            
            if 'min_colors' in filters:
                query += " AND colors_used >= ?"
                params.append(filters['min_colors'])
            
            if 'max_colors' in filters:
                query += " AND colors_used <= ?"
                params.append(filters['max_colors'])
            
            if 'is_optimal' in filters:
                query += " AND is_optimal = ?"
                params.append(int(filters['is_optimal']))
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        results = self.execute_query(query, tuple(params))
        
        result_list = []
        for row in results:
            result = dict(row)
            # Don't parse large JSON fields for bulk queries
            result_list.append(result)
        
        return result_list
    
    def get_best_result_for_puzzle(self, puzzle_id: int) -> Optional[Dict[str, Any]]:
        """
        Get the best result (minimum colors) for a puzzle.
        
        Args:
            puzzle_id: Puzzle ID
            
        Returns:
            Best result dictionary or None
        """
        query = """
        SELECT * FROM results
        WHERE puzzle_id = ? AND is_valid = 1
        ORDER BY colors_used ASC, execution_time ASC
        LIMIT 1
        """
        
        results = self.execute_query(query, (puzzle_id,))
        
        if not results:
            return None
        
        result = dict(results[0])
        if result.get('coloring_result'):
            result['coloring_result'] = json.loads(result['coloring_result'])
        
        return result
    
    # =========================================================================
    # User Evaluation Operations
    # =========================================================================
    
    def save_evaluation(
        self,
        user_id: int,
        evaluation_type: str = 'general',
        pre_test_score: Optional[int] = None,
        post_test_score: Optional[int] = None,
        usability_rating: Optional[int] = None,
        ease_of_use_rating: Optional[int] = None,
        visualization_rating: Optional[int] = None,
        learning_effectiveness: Optional[int] = None,
        would_recommend: Optional[bool] = None,
        time_spent_minutes: Optional[int] = None,
        puzzles_completed: int = 0,
        algorithms_tested: int = 0,
        feedback_text: Optional[str] = None,
        suggestions: Optional[str] = None,
        technical_issues: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """
        Save user evaluation data.
        
        Args:
            user_id: User ID
            evaluation_type: Type of evaluation
            pre_test_score: Pre-test score (0-100)
            post_test_score: Post-test score (0-100)
            usability_rating: Usability rating (1-5)
            ease_of_use_rating: Ease of use rating (1-5)
            visualization_rating: Visualization rating (1-5)
            learning_effectiveness: Learning effectiveness rating (1-5)
            would_recommend: Whether user would recommend
            time_spent_minutes: Time spent in minutes
            puzzles_completed: Number of puzzles completed
            algorithms_tested: Number of algorithms tested
            feedback_text: General feedback
            suggestions: Suggestions for improvement
            technical_issues: Technical issues encountered
            session_id: Session identifier
            
        Returns:
            ID of the created evaluation
        """
        query = """
        INSERT INTO user_evaluations (
            user_id, evaluation_type, pre_test_score, post_test_score,
            usability_rating, ease_of_use_rating, visualization_rating,
            learning_effectiveness, would_recommend, time_spent_minutes,
            puzzles_completed, algorithms_tested, feedback_text,
            suggestions, technical_issues, session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            user_id,
            evaluation_type,
            pre_test_score,
            post_test_score,
            usability_rating,
            ease_of_use_rating,
            visualization_rating,
            learning_effectiveness,
            int(would_recommend) if would_recommend is not None else None,
            time_spent_minutes,
            puzzles_completed,
            algorithms_tested,
            feedback_text,
            suggestions,
            technical_issues,
            session_id
        )
        
        eval_id = self.execute_update(query, params)
        logger.info(f"Saved evaluation {eval_id} for user {user_id}")
        return eval_id
    
    def get_evaluations(
        self, 
        user_id: Optional[int] = None,
        evaluation_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user evaluations.
        
        Args:
            user_id: Optional filter by user ID
            evaluation_type: Optional filter by evaluation type
            
        Returns:
            List of evaluation dictionaries
        """
        query = "SELECT * FROM user_evaluations WHERE 1=1"
        params = []
        
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if evaluation_type:
            query += " AND evaluation_type = ?"
            params.append(evaluation_type)
        
        query += " ORDER BY completed_at DESC"
        
        results = self.execute_query(query, tuple(params) if params else None)
        
        return [dict(row) for row in results]
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated evaluation statistics.
        
        Returns:
            Dictionary with evaluation statistics
        """
        query = """
        SELECT 
            COUNT(*) as total_evaluations,
            AVG(pre_test_score) as avg_pre_score,
            AVG(post_test_score) as avg_post_score,
            AVG(improvement_percentage) as avg_improvement,
            AVG(usability_rating) as avg_usability,
            AVG(ease_of_use_rating) as avg_ease_of_use,
            AVG(visualization_rating) as avg_visualization,
            AVG(learning_effectiveness) as avg_learning,
            SUM(CASE WHEN would_recommend = 1 THEN 1 ELSE 0 END) as recommend_count,
            AVG(time_spent_minutes) as avg_time_spent,
            SUM(puzzles_completed) as total_puzzles_completed,
            SUM(algorithms_tested) as total_algorithms_tested
        FROM user_evaluations
        """
        
        results = self.execute_query(query)
        
        if not results:
            return {}
        
        row = results[0]
        stats = dict(row)
        
        # Round floating point values
        for key, value in stats.items():
            if isinstance(value, float):
                stats[key] = round(value, 2)
        
        # Calculate recommendation percentage
        if stats['total_evaluations'] > 0:
            stats['recommendation_percentage'] = round(
                (stats['recommend_count'] / stats['total_evaluations']) * 100, 2
            )
        
        return stats
    
    # =========================================================================
    # Analytics and Reports
    # =========================================================================
    
    def get_algorithm_performance_summary(
        self, 
        puzzle_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get performance summary for all algorithms.
        
        Args:
            puzzle_type: Optional filter by puzzle type
            
        Returns:
            List of algorithm performance dictionaries
        """
        query = """
        SELECT 
            r.algorithm_name,
            p.puzzle_type,
            COUNT(*) as total_runs,
            AVG(r.colors_used) as avg_colors,
            MIN(r.colors_used) as min_colors,
            MAX(r.colors_used) as max_colors,
            AVG(r.execution_time) as avg_time,
            MIN(r.execution_time) as min_time,
            MAX(r.execution_time) as max_time,
            SUM(CASE WHEN r.is_optimal = 1 THEN 1 ELSE 0 END) as optimal_count
        FROM results r
        JOIN puzzles p ON r.puzzle_id = p.id
        """
        
        if puzzle_type:
            query += " WHERE p.puzzle_type = ?"
            params = (puzzle_type,)
        else:
            params = None
        
        query += " GROUP BY r.algorithm_name, p.puzzle_type ORDER BY avg_colors ASC"
        
        results = self.execute_query(query, params)
        
        summary = []
        for row in results:
            summary.append({
                'algorithm': row['algorithm_name'],
                'puzzle_type': row['puzzle_type'],
                'total_runs': row['total_runs'],
                'avg_colors': round(row['avg_colors'], 2),
                'min_colors': row['min_colors'],
                'max_colors': row['max_colors'],
                'avg_time': round(row['avg_time'], 4),
                'min_time': round(row['min_time'], 4),
                'max_time': round(row['max_time'], 4),
                'optimal_solutions': row['optimal_count']
            })
        
        return summary
    
    def get_database_statistics(self) -> Dict[str, int]:
        """
        Get overall database statistics.
        
        Returns:
            Dictionary with counts of various entities
        """
        stats = {}
        
        # Count tables
        tables = ['users', 'puzzles', 'results', 'user_evaluations']
        
        for table in tables:
            query = f"SELECT COUNT(*) as count FROM {table}"
            result = self.execute_query(query)
            stats[f'{table}_count'] = result[0]['count'] if result else 0
        
        return stats


# =========================================================================
# Convenience Functions
# =========================================================================

def get_database(db_path: str = "graph_coloring.db") -> DatabaseManager:
    """
    Factory function to create and return a DatabaseManager instance.
    
    Args:
        db_path: Path to database file
        
    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_path)


def initialize_database(
    db_path: str = "graph_coloring.db",
    schema_file: str = "database/schema.sql"
) -> DatabaseManager:
    """
    Initialize a new database with schema.
    
    Args:
        db_path: Path to database file
        schema_file: Path to schema SQL file
        
    Returns:
        Initialized DatabaseManager instance
    """
    db = DatabaseManager(db_path)
    db.connect()
    db.initialize_schema(schema_file)
    return db
