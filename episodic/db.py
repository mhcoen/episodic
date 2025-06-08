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

def generate_short_id():
    """
    Generate a short, sequential alphanumeric ID.

    Returns:
        A 2-3 character alphanumeric ID (base-36 encoding)
    """
    with get_connection() as conn:
        c = conn.cursor()
        # Get the highest numeric value of existing short IDs
        c.execute("SELECT MAX(short_id) FROM nodes")
        result = c.fetchone()[0]

        if not result:
            # First node gets '01'
            return '01'

        # Convert from base-36 to integer, increment, convert back
        try:
            value = int(result, 36) + 1
            # Format as base-36, removing '0x' prefix and using lowercase
            new_id = format(value, 'x').zfill(2)

            # If we exceed 2 characters, allow expansion
            return new_id
        except ValueError:
            # If there's an error parsing the existing short_id, start over
            return '01'

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
                short_id TEXT UNIQUE,
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
                # Close the current connection to allow insert_node to create its own
                conn.commit()

                # Create a default root node with an empty string
                root_node_id, root_short_id = insert_node("", None)

                # Return the root node ID
                return root_node_id, root_short_id

    return None

def insert_node(content, parent_id=None):
    node_id = str(uuid.uuid4())
    short_id = generate_short_id()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO nodes (id, short_id, content, parent_id) VALUES (?, ?, ?, ?)",
            (node_id, short_id, content, parent_id)
        )
        conn.commit()
    set_head(node_id)
    return node_id, short_id

def get_node(node_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, short_id, content, parent_id FROM nodes WHERE id = ?", (node_id,))
        row = c.fetchone()
        if not row:
            # Try to find by short_id
            c.execute("SELECT id, short_id, content, parent_id FROM nodes WHERE short_id = ?", (node_id,))
            row = c.fetchone()
    if row:
        return {"id": row[0], "short_id": row[1], "content": row[2], "parent_id": row[3]}
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

def migrate_to_short_ids():
    """
    Add short IDs to existing nodes.

    This function should be called after upgrading to a version that supports short IDs.
    It will add the short_id column to the nodes table if it doesn't exist,
    and generate short IDs for all existing nodes that don't have one.

    Returns:
        The number of nodes that were updated with short IDs
    """
    with get_connection() as conn:
        c = conn.cursor()

        # Check if short_id column exists
        c.execute("PRAGMA table_info(nodes)")
        columns = [info[1] for info in c.fetchall()]

        if 'short_id' not in columns:
            # Add the column
            c.execute("ALTER TABLE nodes ADD COLUMN short_id TEXT")
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_short_id ON nodes(short_id)")

        # Get all nodes without short IDs
        c.execute("SELECT id FROM nodes WHERE short_id IS NULL ORDER BY ROWID")
        nodes = c.fetchall()

        # Assign sequential short IDs
        count = 0

        # Get the highest existing short ID
        c.execute("SELECT MAX(short_id) FROM nodes WHERE short_id IS NOT NULL")
        result = c.fetchone()[0]

        # Start with '01' if no short IDs exist
        next_value = 1

        # If there are existing short IDs, start from the next value
        if result:
            try:
                next_value = int(result, 36) + 1
            except ValueError:
                # If there's an error parsing the existing short_id, start with '01'
                next_value = 1

        # Assign sequential short IDs
        for node_id, in nodes:
            # Format as base-36, removing '0x' prefix and using lowercase
            short_id = format(next_value, 'x').zfill(2)
            next_value += 1

            c.execute("UPDATE nodes SET short_id = ? WHERE id = ?", (short_id, node_id))
            count += 1

        conn.commit()
        return count

def get_recent_nodes(limit=5):
    """
    Get the most recent nodes added to the database.

    Args:
        limit (int): Maximum number of nodes to retrieve

    Returns:
        List of node dictionaries, ordered by recency (most recent first)
    """
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, short_id, content, parent_id
            FROM nodes
            ORDER BY ROWID DESC
            LIMIT ?
        """, (limit,))

        # Get column names from cursor description
        columns = [desc[0] for desc in c.description]
        rows = c.fetchall()

    # Create a list of dictionaries with column names as keys
    result = []
    for row in rows:
        node = {}
        for i, column in enumerate(columns):
            node[column] = row[i]
        result.append(node)

    return result

def get_descendants(node_id):
    """
    Get all descendants of a node.

    Args:
        node_id: ID of the node

    Returns:
        List of IDs of all descendants
    """
    descendants = []

    with get_connection() as conn:
        c = conn.cursor()

        # Use a recursive CTE to find all descendants
        c.execute("""
            WITH RECURSIVE descendants(id) AS (
                SELECT id FROM nodes WHERE parent_id = ?
                UNION ALL
                SELECT n.id FROM nodes n, descendants d WHERE n.parent_id = d.id
            )
            SELECT id FROM descendants
        """, (node_id,))

        descendants = [row[0] for row in c.fetchall()]

    return descendants

def delete_node(node_id):
    """
    Delete a node and all its descendants from the database.

    Args:
        node_id: ID of the node to delete

    Returns:
        List of IDs of all deleted nodes
    """
    # Resolve the node reference if it's not a UUID
    node_id = resolve_node_ref(node_id)

    # Check if the node exists
    node = get_node(node_id)
    if not node:
        return []

    # Get all descendants
    descendants = get_descendants(node_id)

    # Add the node itself to the list of nodes to delete
    nodes_to_delete = [node_id] + descendants

    # Delete all nodes
    with get_connection() as conn:
        c = conn.cursor()

        # Check if the node to delete is the current head
        c.execute("SELECT value FROM meta WHERE key = 'head'")
        head_id = c.fetchone()

        if head_id and head_id[0] in nodes_to_delete:
            # If the head is being deleted, set the parent as the new head
            parent_id = node['parent_id']
            if parent_id:
                set_head(parent_id)
            else:
                # If there's no parent, find another root node
                c.execute("SELECT id FROM nodes WHERE parent_id IS NULL AND id != ? LIMIT 1", (node_id,))
                new_head = c.fetchone()
                if new_head:
                    set_head(new_head[0])
                else:
                    # If there are no other root nodes, remove the head reference
                    c.execute("DELETE FROM meta WHERE key = 'head'")

        # Delete all nodes
        placeholders = ','.join(['?'] * len(nodes_to_delete))
        c.execute(f"DELETE FROM nodes WHERE id IN ({placeholders})", nodes_to_delete)
        conn.commit()

    return nodes_to_delete

def resolve_node_ref(ref):
    """
    Resolve a node reference to its UUID.

    Args:
        ref: A node reference, which can be:
            - A UUID
            - A short ID
            - "HEAD" (case insensitive) or "@head"
            - A relative reference like "HEAD~n"

    Returns:
        The UUID of the referenced node, or None if the reference cannot be resolved
    """
    if not ref:
        return None

    # Handle special references
    if ref == "@head" or ref.upper() == "HEAD" or ref.lower() == "head":
        return get_head()

    # Handle relative references
    if ref.lower().startswith("head~"):
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

    # Check if it's a short ID
    node = get_node(ref)
    if node:
        return node['id']

    # If we get here, assume it's already a UUID
    return ref
