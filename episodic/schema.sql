-- Episodic Database Schema Documentation
-- ======================================
-- This file documents the complete database schema for Episodic.
-- Last updated: 2025-07-01

-- Core conversation storage
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,              -- UUID for the node
    short_id TEXT UNIQUE,             -- Human-readable 2-character ID
    content TEXT NOT NULL,            -- Message content
    parent_id TEXT,                   -- Reference to parent node (NULL for root)
    role TEXT,                        -- 'user', 'assistant', or 'system'
    provider TEXT,                    -- LLM provider (e.g., 'openai', 'anthropic')
    model TEXT,                       -- Model name used for generation
    FOREIGN KEY(parent_id) REFERENCES nodes(id)
);

-- Metadata storage for application state
CREATE TABLE meta (
    key TEXT PRIMARY KEY,             -- Configuration key
    value TEXT NOT NULL               -- JSON-encoded value
);

-- Topic tracking for conversation organization
CREATE TABLE topics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,               -- Topic name (e.g., 'machine-learning')
    start_node_id TEXT NOT NULL,      -- First node in the topic
    end_node_id TEXT,                 -- Last node in the topic (NULL = ongoing)
    confidence TEXT,                  -- Detection confidence level
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(start_node_id) REFERENCES nodes(id),
    FOREIGN KEY(end_node_id) REFERENCES nodes(id)
);

-- Compression tracking for summarized branches
CREATE TABLE compressions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    compressed_node_id TEXT NOT NULL,  -- ID of the compression node
    original_branch_head TEXT NOT NULL, -- Head of the compressed branch
    original_node_count INTEGER NOT NULL,
    original_words INTEGER NOT NULL,
    compressed_words INTEGER NOT NULL,
    compression_ratio REAL NOT NULL,
    strategy TEXT NOT NULL,           -- Compression strategy used
    duration_seconds REAL,            -- Time taken to compress
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(compressed_node_id) REFERENCES nodes(id),
    FOREIGN KEY(original_branch_head) REFERENCES nodes(id)
);

-- Mapping of compressed nodes (for v2 compression system)
CREATE TABLE compression_nodes (
    compression_id TEXT NOT NULL,     -- References compressions.compressed_node_id
    node_id TEXT NOT NULL,            -- Original node that was compressed
    PRIMARY KEY (compression_id, node_id),
    FOREIGN KEY (compression_id) REFERENCES compressions(compressed_node_id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

-- Topic detection scores (unified from manual_index_scores)
CREATE TABLE topic_detection_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_node_short_id TEXT NOT NULL UNIQUE,  -- Node where detection occurred
    window_size INTEGER NOT NULL,
    detection_method TEXT DEFAULT 'sliding_window',  -- 'sliding_window', 'hybrid', 'llm'
    
    -- Window information for sliding window detection
    window_a_start_short_id TEXT,
    window_a_end_short_id TEXT,
    window_a_size INTEGER NOT NULL,
    window_b_start_short_id TEXT,
    window_b_end_short_id TEXT,
    window_b_size INTEGER NOT NULL,
    
    -- Scores
    drift_score REAL NOT NULL,
    keyword_score REAL NOT NULL,
    combined_score REAL NOT NULL,
    
    -- Detection result
    is_boundary BOOLEAN NOT NULL,
    transition_phrase TEXT,           -- Detected transition phrase if any
    
    -- Metadata
    detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    threshold_used REAL
);

-- Schema migration tracking
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,      -- Migration version number
    description TEXT NOT NULL,        -- What the migration does
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_nodes_parent ON nodes(parent_id);
CREATE INDEX idx_nodes_short_id ON nodes(short_id);
CREATE INDEX idx_topics_boundaries ON topics(start_node_id, end_node_id);
CREATE INDEX idx_topics_name ON topics(name);
CREATE INDEX idx_topic_scores_node ON topic_detection_scores(user_node_short_id);
CREATE INDEX idx_topic_scores_boundary ON topic_detection_scores(is_boundary);
CREATE INDEX idx_compressions_node ON compressions(compressed_node_id);

-- Common queries
-- =============

-- Get conversation thread from a node:
-- SELECT * FROM nodes WHERE id IN (
--   WITH RECURSIVE ancestors(id) AS (
--     SELECT id, parent_id FROM nodes WHERE id = ?
--     UNION ALL
--     SELECT n.id, n.parent_id FROM nodes n
--     JOIN ancestors a ON n.id = a.parent_id
--   )
--   SELECT id FROM ancestors
-- ) ORDER BY id;

-- Get all nodes in a topic:
-- WITH RECURSIVE topic_nodes(id) AS (
--   SELECT id, parent_id FROM nodes WHERE id = (SELECT start_node_id FROM topics WHERE id = ?)
--   UNION ALL
--   SELECT n.id, n.parent_id FROM nodes n
--   JOIN topic_nodes t ON n.parent_id = t.id
--   WHERE n.id != (SELECT end_node_id FROM topics WHERE id = ?)
-- )
-- SELECT * FROM nodes WHERE id IN (SELECT id FROM topic_nodes);

-- Find topic boundaries in conversation:
-- SELECT * FROM topic_detection_scores 
-- WHERE is_boundary = 1 
-- ORDER BY detection_timestamp DESC;