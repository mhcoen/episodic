import uuid
import datetime
from typing import Optional

class Node:
    def __init__(self, message: str, parent_id: Optional[str] = None):
        self.id = str(uuid.uuid4())
        self.message = message
        self.timestamp = datetime.datetime.utcnow().isoformat()
        self.parent_id = parent_id

    def to_dict(self):
        return {
            "id": self.id,
            "message": self.message,
            "timestamp": self.timestamp,
            "parent_id": self.parent_id,
        }

class ConversationDAG:
    def __init__(self):
        self.nodes = {}

    def add_node(self, message: str, parent_id: Optional[str] = None) -> Node:
        node = Node(message, parent_id)
        self.nodes[node.id] = node
        return node

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_ancestry(self, node_id: str):
        ancestry = []
        current = self.nodes.get(node_id)
        while current:
            ancestry.append(current)
            current = self.nodes.get(current.parent_id)
        return list(reversed(ancestry))

    def delete_node(self, node_id: str) -> list:
        """
        Delete a node and all its descendants from the DAG.

        Args:
            node_id: ID of the node to delete

        Returns:
            List of IDs of all deleted nodes
        """
        if node_id not in self.nodes:
            return []

        # Find all descendants
        descendants = self._get_descendants(node_id)

        # Add the node itself to the list of nodes to delete
        nodes_to_delete = [node_id] + descendants

        # Delete all nodes
        for node_id in nodes_to_delete:
            if node_id in self.nodes:
                del self.nodes[node_id]

        return nodes_to_delete

    def _get_descendants(self, node_id: str) -> list:
        """
        Get all descendants of a node.

        Args:
            node_id: ID of the node

        Returns:
            List of IDs of all descendants
        """
        descendants = []

        # Find all direct children
        children = [n_id for n_id, node in self.nodes.items() if node.parent_id == node_id]

        # Add children to descendants
        descendants.extend(children)

        # Recursively add descendants of children
        for child_id in children:
            descendants.extend(self._get_descendants(child_id))

        return descendants
