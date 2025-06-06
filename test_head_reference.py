import os
from episodic.db import initialize_db, insert_node, get_node, resolve_node_ref, get_head

# Initialize the database
print("Initializing database...")
initialize_db()

# Add a test node
print("Adding a test node...")
node_id = insert_node("Test message")
print(f"Added node with ID: {node_id}")

# Get the head node ID
head_id = get_head()
print(f"Current HEAD node ID: {head_id}")

# Try to resolve different head references
print("\nTesting different head references:")
print(f"@head -> {resolve_node_ref('@head')}")
print(f"HEAD -> {resolve_node_ref('HEAD')}")
print(f"head -> {resolve_node_ref('head')}")

# Check if the resolved nodes exist
print("\nChecking if resolved nodes exist:")
print(f"@head exists: {get_node(resolve_node_ref('@head')) is not None}")
print(f"HEAD exists: {get_node(resolve_node_ref('HEAD')) is not None}")
print(f"head exists: {get_node(resolve_node_ref('head')) is not None}")