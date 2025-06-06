import sqlite3
import uuid
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "episodic.db"))

def get_connection():
    return sqlite3.connect(DB_PATH)

def database_exists():
    """Check if the database file exists and has tables."""
    if not os.path.exists(DB_PATH):
        return False

    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='nodes'")
            return c.fetchone() is not None
    except sqlite3.Error:
        return False

def initialize_db(erase=False, create_root_node=True):
    """
    Initialize the database.

    Args:
        erase (bool): If True and the database exists, it will be erased.
                     If False and the database exists, it will not be modified.
        create_root_node (bool): If True, creates a default root node if no nodes exist.
    """
    if erase and os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                parent_id TEXT,
                FOREIGN KEY(parent_id) REFERENCES nodes(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()

        # Check if we should create a root node and if there are no existing nodes
        if create_root_node:
            c.execute("SELECT COUNT(*) FROM nodes")
            node_count = c.fetchone()[0]

            if node_count == 0:
                # Create a default root node with an empty string
                root_node_id = str(uuid.uuid4())
                c.execute(
                    "INSERT INTO nodes (id, content, parent_id) VALUES (?, ?, ?)",
                    (root_node_id, "", None)
                )

                # Set this as the head node
                c.execute(
                    "INSERT INTO meta (key, value) VALUES ('head', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (root_node_id,)
                )
                conn.commit()
                return root_node_id

    return None

def insert_node(content, parent_id=None):
    node_id = str(uuid.uuid4())
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO nodes (id, content, parent_id) VALUES (?, ?, ?)", (node_id, content, parent_id))
        conn.commit()
    set_head(node_id)
    return node_id

def get_node(node_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, content, parent_id FROM nodes WHERE id = ?", (node_id,))
        row = c.fetchone()
    if row:
        return {"id": row[0], "content": row[1], "parent_id": row[2]}
    return None

def get_ancestry(node_id):
    ancestry = []
    current_id = node_id
    while current_id:
        node = get_node(current_id)
        if node:
            ancestry.append(node)
            current_id = node["parent_id"]
        else:
            break
    return ancestry[::-1]  # from root to current

def set_head(node_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO meta (key, value) VALUES ('head', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (node_id,))
        conn.commit()

def get_head():
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT value FROM meta WHERE key = 'head'")
        row = c.fetchone()
    if row:
        return row[0]
    return None

def resolve_node_ref(ref):
    if ref == "@head" or ref.upper() == "HEAD" or ref.lower() == "head":
        return get_head()

    if ref.startswith("head~") or ref.startswith("HEAD~"):
        try:
            steps_back = int(ref[len("head~"):])
        except ValueError:
            raise ValueError(f"Invalid relative head reference: {ref}")

        current_id = get_head()
        for _ in range(steps_back):
            node = get_node(current_id)
            if node is None or node['parent_id'] is None:
                return None
            current_id = node['parent_id']
        return current_id

    return ref
