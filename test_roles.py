import os
import sys
from episodic.db import initialize_db, insert_node, get_node, get_ancestry, database_exists, DB_PATH, migrate_to_roles
from episodic.cli import EpisodicShell

def setup():
    """Set up a clean test database with some test nodes."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    
    # Initialize the database
    initialize_db()
    print("Database initialized.")
    
    # Add some test nodes
    root_id, root_short_id = insert_node("System message", None, role="system")
    print(f"Added root node {root_short_id} (UUID: {root_id}) with role 'system'")
    
    user_id, user_short_id = insert_node("User message", root_id, role="user")
    print(f"Added user node {user_short_id} (UUID: {user_id}) with role 'user'")
    
    assistant_id, assistant_short_id = insert_node("Assistant message", user_id, role="assistant")
    print(f"Added assistant node {assistant_short_id} (UUID: {assistant_id}) with role 'assistant'")
    
    return root_id, user_id, assistant_id

def test_roles():
    """Test that roles are stored and retrieved correctly."""
    # Set up the test database
    root_id, user_id, assistant_id = setup()
    
    # Get the nodes and verify the roles
    root_node = get_node(root_id)
    user_node = get_node(user_id)
    assistant_node = get_node(assistant_id)
    
    print("\nVerifying roles:")
    print(f"Root node role: {root_node['role']}")
    print(f"User node role: {user_node['role']}")
    print(f"Assistant node role: {assistant_node['role']}")
    
    # Get the ancestry and verify the roles
    ancestry = get_ancestry(assistant_id)
    
    print("\nVerifying ancestry roles:")
    for i, node in enumerate(ancestry):
        print(f"Node {i} role: {node['role']}")
    
    # Build a conversation context like in handle_chat
    context_messages = []
    for i, node in enumerate(ancestry):
        # Skip the first node if it's a system message or has no parent
        if i == 0 and node['parent_id'] is None:
            continue
        
        # Use the stored role if available, otherwise fall back to alternating roles
        role = node.get('role')
        if role is None:
            # Fallback to alternating roles if role is not stored
            role = "user" if i % 2 == 0 else "assistant"
        context_messages.append({"role": role, "content": node['content']})
    
    print("\nVerifying context messages:")
    for i, msg in enumerate(context_messages):
        print(f"Message {i} role: {msg['role']}, content: {msg['content']}")
    
    # Test the migration function
    print("\nTesting migration function:")
    # Remove the role column from the nodes
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Create a temporary table without the role column
        c.execute("CREATE TABLE nodes_temp (id TEXT PRIMARY KEY, short_id TEXT UNIQUE, content TEXT NOT NULL, parent_id TEXT, FOREIGN KEY(parent_id) REFERENCES nodes(id))")
        # Copy data from nodes to nodes_temp
        c.execute("INSERT INTO nodes_temp SELECT id, short_id, content, parent_id FROM nodes")
        # Drop the nodes table
        c.execute("DROP TABLE nodes")
        # Rename nodes_temp to nodes
        c.execute("ALTER TABLE nodes_temp RENAME TO nodes")
        conn.commit()
    
    # Run the migration function
    count = migrate_to_roles()
    print(f"Migrated {count} nodes")
    
    # Get the nodes and verify the roles
    root_node = get_node(root_id)
    user_node = get_node(user_id)
    assistant_node = get_node(assistant_id)
    
    print("\nVerifying roles after migration:")
    print(f"Root node role: {root_node['role']}")
    print(f"User node role: {user_node['role']}")
    print(f"Assistant node role: {assistant_node['role']}")

if __name__ == "__main__":
    test_roles()