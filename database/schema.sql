-- ============================================================================
-- Database Schema for Graph Coloring Research Platform
-- ============================================================================
-- Created: December 2025
-- Database: SQLite
-- Purpose: Store users, puzzles, algorithm results, and user evaluations
-- ============================================================================

-- ============================================================================
-- Table: users
-- Description: Stores user account information
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT,
    full_name TEXT,
    role TEXT DEFAULT 'user' CHECK(role IN ('user', 'researcher', 'admin')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    is_active BOOLEAN DEFAULT 1,
    CONSTRAINT chk_email_format CHECK (email LIKE '%_@__%.__%')
);

-- Index for faster user lookups
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- ============================================================================
-- Table: puzzles
-- Description: Stores graph puzzles created by users
-- ============================================================================
CREATE TABLE IF NOT EXISTS puzzles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    puzzle_type TEXT NOT NULL CHECK(puzzle_type IN (
        'sudoku', 'nqueens', 'kakuro', 'futoshiki', 
        'latin_square', 'map_coloring', 'custom_graph'
    )),
    difficulty_level TEXT CHECK(difficulty_level IN ('easy', 'medium', 'hard', 'expert')),
    num_vertices INTEGER NOT NULL CHECK(num_vertices > 0),
    num_edges INTEGER NOT NULL CHECK(num_edges >= 0),
    density REAL CHECK(density >= 0.0 AND density <= 1.0),
    chromatic_number INTEGER,
    graph_data TEXT NOT NULL, -- JSON format: {vertices: [], edges: [], coordinates: {}}
    initial_constraints TEXT, -- JSON format for pre-filled values or constraints
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_public BOOLEAN DEFAULT 0,
    is_verified BOOLEAN DEFAULT 0,
    views_count INTEGER DEFAULT 0,
    attempts_count INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    CONSTRAINT chk_edges_vertices CHECK(num_edges <= (num_vertices * (num_vertices - 1) / 2))
);

-- Indexes for faster puzzle queries
CREATE INDEX IF NOT EXISTS idx_puzzles_user_id ON puzzles(user_id);
CREATE INDEX IF NOT EXISTS idx_puzzles_type ON puzzles(puzzle_type);
CREATE INDEX IF NOT EXISTS idx_puzzles_difficulty ON puzzles(difficulty_level);
CREATE INDEX IF NOT EXISTS idx_puzzles_public ON puzzles(is_public);
CREATE INDEX IF NOT EXISTS idx_puzzles_created_at ON puzzles(created_at);
CREATE INDEX IF NOT EXISTS idx_puzzles_chromatic ON puzzles(chromatic_number);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_puzzles_type_public ON puzzles(puzzle_type, is_public);

-- ============================================================================
-- Table: results
-- Description: Stores algorithm execution results for puzzles
-- ============================================================================
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    puzzle_id INTEGER NOT NULL,
    user_id INTEGER,
    algorithm_name TEXT NOT NULL CHECK(algorithm_name IN (
        'greedy', 'dsatur', 'welsh_powell', 'tabu_search', 
        'dqn', 'backtracking', 'genetic', 'simulated_annealing'
    )),
    colors_used INTEGER NOT NULL CHECK(colors_used > 0),
    execution_time REAL NOT NULL CHECK(execution_time >= 0),
    memory_usage REAL, -- in MB
    iterations INTEGER,
    is_optimal BOOLEAN DEFAULT 0,
    is_valid BOOLEAN DEFAULT 1,
    coloring_result TEXT NOT NULL, -- JSON format: {node: color, ...}
    algorithm_params TEXT, -- JSON format for algorithm parameters
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (puzzle_id) REFERENCES puzzles(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Indexes for result queries and analysis
CREATE INDEX IF NOT EXISTS idx_results_puzzle_id ON results(puzzle_id);
CREATE INDEX IF NOT EXISTS idx_results_user_id ON results(user_id);
CREATE INDEX IF NOT EXISTS idx_results_algorithm ON results(algorithm_name);
CREATE INDEX IF NOT EXISTS idx_results_colors_used ON results(colors_used);
CREATE INDEX IF NOT EXISTS idx_results_execution_time ON results(execution_time);
CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp);
CREATE INDEX IF NOT EXISTS idx_results_optimal ON results(is_optimal);

-- Composite indexes for performance analysis
CREATE INDEX IF NOT EXISTS idx_results_puzzle_algo ON results(puzzle_id, algorithm_name);
CREATE INDEX IF NOT EXISTS idx_results_algo_colors ON results(algorithm_name, colors_used);

-- ============================================================================
-- Table: user_evaluations
-- Description: Stores user study data and feedback
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    evaluation_type TEXT DEFAULT 'general' CHECK(evaluation_type IN (
        'general', 'pre_study', 'post_study', 'usability', 'algorithm_comparison'
    )),
    pre_test_score INTEGER CHECK(pre_test_score >= 0 AND pre_test_score <= 100),
    post_test_score INTEGER CHECK(post_test_score >= 0 AND post_test_score <= 100),
    improvement_percentage REAL,
    usability_rating INTEGER CHECK(usability_rating >= 1 AND usability_rating <= 5),
    ease_of_use_rating INTEGER CHECK(ease_of_use_rating >= 1 AND ease_of_use_rating <= 5),
    visualization_rating INTEGER CHECK(visualization_rating >= 1 AND visualization_rating <= 5),
    learning_effectiveness INTEGER CHECK(learning_effectiveness >= 1 AND learning_effectiveness <= 5),
    would_recommend BOOLEAN,
    time_spent_minutes INTEGER CHECK(time_spent_minutes >= 0),
    puzzles_completed INTEGER DEFAULT 0,
    algorithms_tested INTEGER DEFAULT 0,
    feedback_text TEXT,
    suggestions TEXT,
    technical_issues TEXT,
    completed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Indexes for evaluation analysis
CREATE INDEX IF NOT EXISTS idx_evaluations_user_id ON user_evaluations(user_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_type ON user_evaluations(evaluation_type);
CREATE INDEX IF NOT EXISTS idx_evaluations_completed_at ON user_evaluations(completed_at);
CREATE INDEX IF NOT EXISTS idx_evaluations_rating ON user_evaluations(usability_rating);

-- ============================================================================
-- Table: algorithm_comparisons
-- Description: Stores comparative analysis between algorithms
-- ============================================================================
CREATE TABLE IF NOT EXISTS algorithm_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    puzzle_id INTEGER NOT NULL,
    user_id INTEGER,
    algorithms_compared TEXT NOT NULL, -- JSON array of algorithm names
    best_algorithm TEXT,
    best_colors INTEGER,
    best_time REAL,
    comparison_data TEXT, -- JSON format with detailed comparison
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (puzzle_id) REFERENCES puzzles(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_comparisons_puzzle_id ON algorithm_comparisons(puzzle_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_user_id ON algorithm_comparisons(user_id);

-- ============================================================================
-- Table: experiment_sessions
-- Description: Tracks research experiment sessions
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiment_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_name TEXT NOT NULL,
    description TEXT,
    user_id INTEGER,
    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    end_time DATETIME,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'completed', 'cancelled')),
    total_puzzles INTEGER DEFAULT 0,
    total_results INTEGER DEFAULT 0,
    session_data TEXT, -- JSON format for session metadata
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON experiment_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON experiment_sessions(status);

-- ============================================================================
-- Table: algorithm_performance_metrics
-- Description: Aggregated performance metrics for algorithms
-- ============================================================================
CREATE TABLE IF NOT EXISTS algorithm_performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm_name TEXT NOT NULL,
    puzzle_type TEXT NOT NULL,
    total_runs INTEGER DEFAULT 0,
    avg_colors REAL,
    min_colors INTEGER,
    max_colors INTEGER,
    avg_execution_time REAL,
    min_execution_time REAL,
    max_execution_time REAL,
    success_rate REAL CHECK(success_rate >= 0.0 AND success_rate <= 1.0),
    optimal_solutions_count INTEGER DEFAULT 0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(algorithm_name, puzzle_type)
);

CREATE INDEX IF NOT EXISTS idx_perf_algorithm ON algorithm_performance_metrics(algorithm_name);
CREATE INDEX IF NOT EXISTS idx_perf_puzzle_type ON algorithm_performance_metrics(puzzle_type);

-- ============================================================================
-- Triggers
-- ============================================================================

-- Trigger to update puzzles.updated_at on modification
CREATE TRIGGER IF NOT EXISTS update_puzzle_timestamp 
AFTER UPDATE ON puzzles
FOR EACH ROW
BEGIN
    UPDATE puzzles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger to increment puzzle attempts count
CREATE TRIGGER IF NOT EXISTS increment_puzzle_attempts
AFTER INSERT ON results
FOR EACH ROW
BEGIN
    UPDATE puzzles SET attempts_count = attempts_count + 1 WHERE id = NEW.puzzle_id;
END;

-- Trigger to calculate improvement percentage in evaluations
CREATE TRIGGER IF NOT EXISTS calculate_improvement
AFTER INSERT ON user_evaluations
FOR EACH ROW
WHEN NEW.pre_test_score IS NOT NULL AND NEW.post_test_score IS NOT NULL
BEGIN
    UPDATE user_evaluations 
    SET improvement_percentage = 
        CASE 
            WHEN NEW.pre_test_score > 0 
            THEN ((NEW.post_test_score - NEW.pre_test_score) * 100.0 / NEW.pre_test_score)
            ELSE NULL
        END
    WHERE id = NEW.id;
END;

-- ============================================================================
-- Views
-- ============================================================================

-- View: Top performing algorithms by puzzle type
CREATE VIEW IF NOT EXISTS v_top_algorithms AS
SELECT 
    puzzle_type,
    algorithm_name,
    AVG(colors_used) as avg_colors,
    AVG(execution_time) as avg_time,
    COUNT(*) as total_runs,
    SUM(CASE WHEN is_optimal = 1 THEN 1 ELSE 0 END) as optimal_count
FROM results r
JOIN puzzles p ON r.puzzle_id = p.id
GROUP BY puzzle_type, algorithm_name;

-- View: User statistics
CREATE VIEW IF NOT EXISTS v_user_stats AS
SELECT 
    u.id,
    u.username,
    COUNT(DISTINCT p.id) as puzzles_created,
    COUNT(DISTINCT r.id) as results_generated,
    COUNT(DISTINCT e.id) as evaluations_completed,
    AVG(e.usability_rating) as avg_rating
FROM users u
LEFT JOIN puzzles p ON u.id = p.user_id
LEFT JOIN results r ON u.id = r.user_id
LEFT JOIN user_evaluations e ON u.id = e.user_id
GROUP BY u.id, u.username;

-- View: Puzzle difficulty analysis
CREATE VIEW IF NOT EXISTS v_puzzle_difficulty AS
SELECT 
    p.id,
    p.name,
    p.puzzle_type,
    p.num_vertices,
    p.num_edges,
    p.density,
    p.chromatic_number,
    AVG(r.colors_used) as avg_colors_solved,
    MIN(r.colors_used) as best_colors,
    AVG(r.execution_time) as avg_solve_time,
    COUNT(r.id) as solve_attempts
FROM puzzles p
LEFT JOIN results r ON p.id = r.puzzle_id
GROUP BY p.id;

-- ============================================================================
-- Initial Data (Optional)
-- ============================================================================

-- Insert a default admin user (password should be hashed in production)
INSERT OR IGNORE INTO users (id, username, email, role, is_active) 
VALUES (1, 'admin', 'admin@graphcoloring.com', 'admin', 1);

-- ============================================================================
-- End of Schema
-- ============================================================================
