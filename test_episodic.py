# Import necessary functions from the episodic.db module
from episodic.db import initialize_db, insert_node, get_node, get_ancestry, set_head, get_head, resolve_node_ref
import uuid

# Utility function to display node information in a readable format
def print_node(node):
    """
    Prints the details of a node (ID, parent ID, and content) if the node exists.
    Otherwise, prints a "Node not found" message.
    """
    if node:
        print(f"Node ID: {node['id']}")
        print(f"Parent: {node['parent_id']}")
        print(f"Message: {node['content']}")
    else:
        print("Node not found.")

# Utility function to display the ancestry (chain of nodes) for a given node
def print_ancestry(node_id):
    """
    Retrieves and prints the ancestry (chain of nodes from root to the specified node)
    for a given node ID. Each node is displayed with its ID and content.
    """
    ancestry = get_ancestry(node_id)
    for node in ancestry:
        print(f"{node['id']}: {node['content']}")

# Main test function that demonstrates the functionality of the episodic database
def run_tests():
    """
    Runs a series of tests to demonstrate the functionality of the episodic database system.
    This includes initializing the database, adding nodes, and navigating through the node hierarchy.
    """
    # Initialize the database (creates tables if they don't exist)
    print("== Initializing DB ==")
    initialize_db()

    # Create the first node with no parent (root node)
    print("\n== Adding first node ==")
    node1_id = insert_node("This is the first message.")
    print(f"Added node {node1_id}")

    # Display the current head node (should be the first node we just added)
    print("\n== Showing @head ==")
    node = get_node(get_head())
    print_node(node)

    # Create a second node with the first node as its parent
    print("\n== Adding second node ==")
    node2_id = insert_node("This is the second message.", parent_id=node1_id)
    print(f"Added node {node2_id}")

    # Display the current head node (should now be the second node)
    print("\n== Showing @head ==")
    node = get_node(get_head())
    print_node(node)

    # Create a third node with the second node as its parent
    print("\n== Adding third node ==")
    node3_id = insert_node("Third message in the chain.", parent_id=node2_id)
    print(f"Added node {node3_id}")

    # Display the ancestry (chain of nodes) for the current head node
    # This should show all three nodes from root to head
    print("\n== Showing ancestry of @head ==")
    resolved_id = resolve_node_ref("@head")
    print_ancestry(resolved_id)

    # Demonstrate relative node referencing: head~1 refers to the parent of the current head
    # This should show the second node
    print("\n== Showing head~1 ==")
    resolved_id = resolve_node_ref("head~1")
    print_node(get_node(resolved_id))

    # Demonstrate relative node referencing: head~2 refers to the grandparent of the current head
    # This should show the first node
    print("\n== Showing head~2 ==")
    resolved_id = resolve_node_ref("head~2")
    print_node(get_node(resolved_id))

    # Test error handling for invalid relative references
    # Attempting to go back 99 generations should fail as we only have 3 nodes
    print("\n== Attempt to show head~99 (should fail) ==")
    try:
        resolved_id = resolve_node_ref("head~99")
        print_node(get_node(resolved_id))
    except Exception as e:
        print("Expected failure:", e)

# Execute the tests when the script is run directly
if __name__ == "__main__":
    run_tests()
