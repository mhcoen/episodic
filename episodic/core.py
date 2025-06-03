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
    
