"""
Test fixtures for conversation data.

Provides reusable test conversations with known topic boundaries
for testing topic detection and other conversation features.
"""

from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import uuid


def create_test_node(
    content: str,
    role: str = "user",
    parent_id: str = None,
    timestamp: datetime = None,
    short_id: str = None
) -> Dict:
    """Create a test node with given properties."""
    node_id = str(uuid.uuid4())
    if not timestamp:
        timestamp = datetime.now()
    if not short_id:
        short_id = f"n{node_id[:2]}"
    
    return {
        'id': node_id,
        'short_id': short_id,
        'message': content,
        'role': role,
        'parent_id': parent_id,
        'timestamp': timestamp.isoformat(),
        'model_name': 'test-model',
        'system_prompt': 'Test system prompt',
        'response': content if role == "assistant" else None
    }


def create_test_conversation(
    topic_messages: List[Tuple[str, List[str]]]
) -> Tuple[List[Dict], List[int]]:
    """
    Create a test conversation with known topic boundaries.
    
    Args:
        topic_messages: List of (topic_name, [messages]) tuples
        
    Returns:
        Tuple of (nodes, topic_boundaries) where topic_boundaries
        are the indices where new topics start
    """
    nodes = []
    topic_boundaries = []
    parent_id = None
    timestamp = datetime.now()
    
    for topic_idx, (topic_name, messages) in enumerate(topic_messages):
        # Mark topic boundary
        if nodes:
            topic_boundaries.append(len(nodes))
            
        for msg_idx, message in enumerate(messages):
            # Alternate between user and assistant
            role = "user" if msg_idx % 2 == 0 else "assistant"
            
            node = create_test_node(
                content=message,
                role=role,
                parent_id=parent_id,
                timestamp=timestamp
            )
            
            nodes.append(node)
            parent_id = node['id']
            timestamp += timedelta(seconds=30)
    
    return nodes, topic_boundaries


# Pre-defined test conversations
THREE_TOPICS_CONVERSATION = create_test_conversation([
    ("Mars Colonization", [
        "Tell me about the challenges of colonizing Mars",
        "Mars colonization faces several major challenges including radiation exposure, atmospheric differences, and resource scarcity...",
        "What about the psychological challenges?",
        "The psychological challenges are significant. Isolation, confinement, and distance from Earth can cause severe mental health issues...",
        "How long would the journey take?",
        "The journey to Mars typically takes 6-9 months depending on the alignment of Earth and Mars...",
    ]),
    ("Italian Cooking", [
        "I want to learn how to make authentic Italian pasta",
        "Making authentic Italian pasta starts with using the right ingredients - tipo 00 flour and fresh eggs...",
        "What's the proper way to cook it?",
        "The key to cooking pasta properly is using plenty of salted water - at least 4-6 quarts per pound...",
        "Tell me about different pasta shapes",
        "Italian cuisine features hundreds of pasta shapes, each designed for specific sauces and preparations...",
    ]),
    ("Neural Networks", [
        "Explain how neural networks learn",
        "Neural networks learn through a process called backpropagation, where errors are propagated backwards...",
        "What are activation functions?",
        "Activation functions introduce non-linearity into neural networks, allowing them to learn complex patterns...",
        "How do convolutional layers work?",
        "Convolutional layers apply filters across input data to detect features like edges, shapes, and patterns...",
    ])
])


GRADUAL_DRIFT_CONVERSATION = create_test_conversation([
    ("Programming Basics", [
        "What's the difference between a variable and a constant?",
        "A variable can change its value during program execution, while a constant maintains the same value...",
        "Can you show me an example in Python?",
        "In Python, variables are simply assigned: x = 5. Constants are by convention written in uppercase: MAX_SIZE = 100...",
        "What about data types in Python?",
        "Python has several built-in data types including int, float, str, list, dict, and tuple...",
        "How does Python handle memory management?",  # Starting to drift
        "Python uses automatic memory management with garbage collection. It uses reference counting...",
    ]),
    ("Python Advanced Topics", [
        "Tell me about Python's garbage collector",
        "Python's garbage collector uses generational garbage collection to handle circular references...",
        "What are decorators in Python?",
        "Decorators are a powerful feature that allow you to modify or enhance functions without changing their code...",
        "How do metaclasses work?",
        "Metaclasses are classes whose instances are classes. They control class creation and behavior...",
    ])
])


SINGLE_TOPIC_CONVERSATION = create_test_conversation([
    ("Machine Learning", [
        "What is machine learning?",
        "Machine learning is a subset of AI that enables systems to learn from data without explicit programming...",
        "What's the difference between supervised and unsupervised learning?",
        "Supervised learning uses labeled data to train models, while unsupervised learning finds patterns in unlabeled data...",
        "Can you explain overfitting?",
        "Overfitting occurs when a model learns the training data too well, including noise and outliers...",
        "What are some ways to prevent overfitting?",
        "Common techniques include regularization, cross-validation, dropout, and using more training data...",
        "Tell me about neural networks",
        "Neural networks are computing systems inspired by biological neural networks in animal brains...",
        "What's deep learning?",
        "Deep learning uses neural networks with multiple layers to progressively extract higher-level features...",
    ])
])


def assert_topics_detected(
    detected_topics: List[Dict],
    expected_count: int,
    expected_names: List[str] = None
):
    """
    Helper to assert topic detection results.
    
    Args:
        detected_topics: List of detected topic dictionaries
        expected_count: Expected number of topics
        expected_names: Optional list of expected topic names (fuzzy match)
    """
    assert len(detected_topics) == expected_count, \
        f"Expected {expected_count} topics, got {len(detected_topics)}"
    
    if expected_names:
        detected_names = [t['name'].lower() for t in detected_topics]
        for expected_name in expected_names:
            # Fuzzy match - check if any detected name contains expected keywords
            keywords = expected_name.lower().split()
            found = any(
                all(keyword in name for keyword in keywords)
                for name in detected_names
            )
            assert found, f"Expected topic containing '{expected_name}' not found in {detected_names}"


def get_test_messages_only(nodes: List[Dict], role: str = "user") -> List[str]:
    """Extract messages of a specific role from nodes."""
    return [node['message'] for node in nodes if node['role'] == role]