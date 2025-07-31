#!/usr/bin/env python3
"""
Retroactively create topics using the improved (4,2) window detection.
This version recalculates drift using the better approach we discovered.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple

# Configuration
WINDOW_CONFIG = {
    'before_window': -4,  # Last 4 messages
    'after_window': 2,    # Next 2 messages (user + assistant)
    'threshold': 0.25     # From our testing, optimal for (4,2)
}
MIN_TOPIC_LENGTH = 3      # Minimum messages per topic

def get_db_connection():
    """Get database connection."""
    db_path = Path.home() / '.episodic' / 'episodic.db'
    return sqlite3.connect(str(db_path))

def load_all_messages():
    """Load all messages from the database in order."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, short_id, role, content, created_at, parent_id
        FROM nodes
        ORDER BY created_at
    """)
    
    messages = []
    for row in cursor.fetchall():
        messages.append({
            'id': row[0],
            'short_id': row[1],
            'role': row[2],
            'content': row[3],
            'created_at': row[4],
            'parent_id': row[5]
        })
    
    conn.close()
    return messages

def get_sentence_embedding(text, model, tokenizer, device):
    """Get sentence embedding using mean pooling."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        
    return mean_pooled[0].cpu().numpy()

def calculate_window_similarity(messages: List[Dict], idx: int, model, tokenizer, device) -> float:
    """Calculate similarity using (4,2) window approach."""
    
    # Get window boundaries
    before_start = max(0, idx + WINDOW_CONFIG['before_window'])
    before_end = idx
    after_start = idx
    after_end = min(len(messages), idx + WINDOW_CONFIG['after_window'])
    
    # Not enough context for a proper window
    if before_end - before_start < 2 or after_end - after_start < 1:
        return 1.0  # High similarity (no boundary)
    
    # Get messages for each window
    before_messages = messages[before_start:before_end]
    after_messages = messages[after_start:after_end]
    
    # Convert to text
    before_text = " ".join([
        f"{msg['role']}: {msg['content']}" for msg in before_messages
    ])
    after_text = " ".join([
        f"{msg['role']}: {msg['content']}" for msg in after_messages
    ])
    
    # Get embeddings
    before_emb = get_sentence_embedding(before_text, model, tokenizer, device)
    after_emb = get_sentence_embedding(after_text, model, tokenizer, device)
    
    # Calculate cosine similarity
    similarity = 1 - cosine(before_emb, after_emb)
    return similarity

def detect_topic_boundaries(messages: List[Dict]) -> List[Tuple[int, float]]:
    """Detect topic boundaries using (4,2) window approach."""
    
    print("Loading embedding model...")
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    model_name = "sentence-transformers/all-mpnet-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    boundaries = []
    
    print(f"Analyzing {len(messages)} messages with (4,2) window...")
    
    # Check each potential boundary point
    for i in range(len(messages)):
        # Only check at user messages (potential topic starts)
        if messages[i]['role'] != 'user':
            continue
            
        # Skip if too close to start
        if i < 4:
            continue
            
        similarity = calculate_window_similarity(messages, i, model, tokenizer, device)
        
        # Lower similarity means topic boundary
        if similarity < WINDOW_CONFIG['threshold']:
            boundaries.append((i, similarity))
            print(f"  Boundary detected at {messages[i]['short_id']}: similarity={similarity:.3f}")
    
    return boundaries

def create_topics_from_boundaries(messages: List[Dict], boundaries: List[Tuple[int, float]]) -> List[Dict]:
    """Create topic definitions from detected boundaries."""
    
    topics = []
    
    # Start with first message
    topic_start = 0
    
    for boundary_idx, similarity in boundaries:
        # End previous topic at the message before the boundary
        topic_end = boundary_idx - 1
        
        if topic_end - topic_start >= MIN_TOPIC_LENGTH:
            topics.append({
                'start_idx': topic_start,
                'end_idx': topic_end,
                'start_node': messages[topic_start],
                'end_node': messages[topic_end],
                'boundary_similarity': similarity,
                'message_count': topic_end - topic_start + 1
            })
        
        # Start new topic at the boundary
        topic_start = boundary_idx
    
    # Don't forget the last topic
    if topic_start < len(messages) - 1:
        topics.append({
            'start_idx': topic_start,
            'end_idx': len(messages) - 1,
            'start_node': messages[topic_start],
            'end_node': messages[-1],
            'boundary_similarity': None,
            'message_count': len(messages) - topic_start
        })
    
    return topics

def suggest_topic_name(messages: List[Dict], start_idx: int, end_idx: int) -> str:
    """Suggest a topic name based on message content."""
    
    topic_messages = messages[start_idx:end_idx + 1]
    user_messages = [m for m in topic_messages if m['role'] == 'user']
    
    if not user_messages:
        return "Unknown Topic"
    
    # Analyze first few user messages for keywords
    combined_text = " ".join([m['content'].lower() for m in user_messages[:3]])
    
    # Keyword-based naming
    if 'csv' in combined_text and 'python' in combined_text:
        return "Python CSV Processing"
    elif 'pandas' in combined_text or 'dataframe' in combined_text:
        return "Pandas Data Analysis"
    elif 'carbonara' in combined_text or 'pasta' in combined_text:
        return "Italian Cooking"
    elif 'machine learning' in combined_text or 'supervised' in combined_text:
        return "Machine Learning"
    elif 'japan' in combined_text or 'tokyo' in combined_text:
        return "Japan Travel"
    elif 'react' in combined_text or 'javascript' in combined_text:
        return "Web Development"
    elif 'workout' in combined_text or 'exercise' in combined_text:
        return "Fitness"
    elif 'garden' in combined_text or 'vegetable' in combined_text:
        return "Gardening"
    elif 'budget' in combined_text or 'invest' in combined_text:
        return "Personal Finance"
    else:
        # Use first few words of first message
        first_words = user_messages[0]['content'].split()[:3]
        return " ".join(first_words).title()

def main():
    """Main execution."""
    print("Retroactive Topic Detection using (4,2) Window Approach")
    print("=" * 60)
    print(f"Window: {WINDOW_CONFIG['before_window']} messages before, {WINDOW_CONFIG['after_window']} after")
    print(f"Threshold: {WINDOW_CONFIG['threshold']}")
    print()
    
    # Load messages
    messages = load_all_messages()
    print(f"Loaded {len(messages)} messages")
    
    # Detect boundaries
    boundaries = detect_topic_boundaries(messages)
    print(f"\nFound {len(boundaries)} topic boundaries")
    
    # Create topics
    topics = create_topics_from_boundaries(messages, boundaries)
    print(f"Created {len(topics)} topics")
    
    # Display results
    print("\n" + "=" * 60)
    print("DETECTED TOPICS:")
    print("=" * 60)
    
    for i, topic in enumerate(topics):
        name = suggest_topic_name(messages, topic['start_idx'], topic['end_idx'])
        
        print(f"\nTopic {i + 1}: {name}")
        print(f"  Messages: {topic['message_count']}")
        print(f"  Range: {topic['start_node']['short_id']} → {topic['end_node']['short_id']}")
        
        if topic['boundary_similarity'] is not None:
            print(f"  Boundary similarity: {topic['boundary_similarity']:.3f}")
        
        # Show first user message
        first_user_msg = next((messages[j]['content'] for j in range(topic['start_idx'], topic['end_idx'] + 1) 
                              if messages[j]['role'] == 'user'), "N/A")
        print(f"  First question: {first_user_msg[:60]}...")
    
    # Generate SQL
    print("\n" + "=" * 60)
    print("SQL TO CREATE TOPICS:")
    print("=" * 60)
    
    for i, topic in enumerate(topics):
        name = suggest_topic_name(messages, topic['start_idx'], topic['end_idx'])
        topic_id = f"retro_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        sql = f"""
INSERT INTO topics (id, name, start_node_id, end_node_id, status, created_at)
VALUES ('{topic_id}', '{name}', '{topic['start_node']['id']}', 
        '{topic['end_node']['id']}', 'completed', CURRENT_TIMESTAMP);"""
        print(sql)
    
    # Save to file
    with open("retroactive_topics_v2.sql", 'w') as f:
        f.write("-- Retroactive topics using (4,2) window detection\n")
        f.write(f"-- Generated: {datetime.now()}\n")
        f.write(f"-- Threshold: {WINDOW_CONFIG['threshold']}\n\n")
        
        for i, topic in enumerate(topics):
            name = suggest_topic_name(messages, topic['start_idx'], topic['end_idx'])
            topic_id = f"retro_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            sql = f"""INSERT INTO topics (id, name, start_node_id, end_node_id, status, created_at)
VALUES ('{topic_id}', '{name}', '{topic['start_node']['id']}', '{topic['end_node']['id']}', 'completed', CURRENT_TIMESTAMP);
"""
            f.write(sql)
    
    print(f"\nSQL saved to: retroactive_topics_v2.sql")

if __name__ == "__main__":
    main()