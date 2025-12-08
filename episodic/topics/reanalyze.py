"""
Topic reanalysis using hierarchical contiguous clustering.

This module provides retroactive topic detection by analyzing the complete
conversation and finding optimal segment boundaries using hierarchical
agglomerative clustering with a contiguity constraint.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import typer

from episodic.config import config
from episodic.ml.drift import ConversationalDrift
from episodic.db import get_head, get_ancestry, get_node
from episodic.db_topics import store_topic, get_all_topics
from episodic.configuration import get_system_color, get_heading_color, get_text_color

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A contiguous segment of messages."""
    start_idx: int
    end_idx: int
    node_ids: List[str]
    embedding: Optional[List[float]] = None


def get_conversation_messages() -> List[Dict[str, Any]]:
    """
    Get all messages in the current conversation in chronological order.

    Returns:
        List of message dicts with 'content', 'role', 'id', 'short_id'
    """
    head = get_head()
    if not head:
        return []

    # Get ancestry from root to head
    ancestry = get_ancestry(head)

    # Filter to user messages only (user messages drive topic changes)
    messages = []
    for node in ancestry:
        role = node.get('role', '')
        if role == 'user':
            messages.append({
                'content': node.get('content', ''),
                'role': role,
                'id': node.get('id'),
                'short_id': node.get('short_id')
            })

    return messages


def compute_embeddings(
    messages: List[Dict[str, Any]],
    drift_calc: ConversationalDrift
) -> List[List[float]]:
    """
    Compute embeddings for all messages.

    Returns:
        List of embedding vectors, one per message
    """
    embeddings = []
    for msg in messages:
        text = f"{msg['role']}: {msg['content']}"
        emb = drift_calc.embedding_provider.embed(text)
        embeddings.append(emb)
    return embeddings


def compute_segment_embedding(
    embeddings: List[List[float]],
    start_idx: int,
    end_idx: int
) -> List[float]:
    """
    Compute the centroid embedding for a segment.

    Args:
        embeddings: All message embeddings
        start_idx: Start index (inclusive)
        end_idx: End index (inclusive)

    Returns:
        Centroid embedding vector
    """
    segment_embeddings = embeddings[start_idx:end_idx + 1]
    if not segment_embeddings:
        return []

    # Compute centroid (mean of all embeddings)
    dim = len(segment_embeddings[0])
    centroid = [0.0] * dim
    for emb in segment_embeddings:
        for i, val in enumerate(emb):
            centroid[i] += val

    n = len(segment_embeddings)
    centroid = [v / n for v in centroid]
    return centroid


def compute_similarity(emb1: List[float], emb2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.

    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if not emb1 or not emb2:
        return 0.0

    dot = sum(a * b for a, b in zip(emb1, emb2))
    norm1 = sum(a * a for a in emb1) ** 0.5
    norm2 = sum(b * b for b in emb2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


@dataclass
class MergeStep:
    """Record of a merge operation."""
    num_segments: int
    similarity: float
    segments_snapshot: List[Segment]


def hierarchical_segment(
    messages: List[Dict[str, Any]],
    embeddings: List[List[float]],
    min_similarity: float = 0.0,
    min_segments: int = 1,
    use_elbow: bool = True,
    verbose: bool = False
) -> List[Segment]:
    """
    Perform hierarchical agglomerative clustering with contiguity constraint.

    Algorithm:
    1. Start with each message as its own segment
    2. Compute similarity between adjacent segment pairs
    3. Merge the most similar adjacent pair
    4. Track similarity at each step to find natural breakpoints
    5. Use elbow detection to find optimal stopping point

    Args:
        messages: List of message dicts
        embeddings: List of embedding vectors
        min_similarity: Stop merging when best similarity drops below this (ignored if use_elbow)
        min_segments: Minimum number of segments to keep
        use_elbow: If True, automatically detect optimal stopping point
        verbose: Print progress

    Returns:
        List of Segment objects representing final topic segments
    """
    if not messages:
        return []

    n = len(messages)

    # Initialize: each message is its own segment
    segments = []
    for i in range(n):
        seg = Segment(
            start_idx=i,
            end_idx=i,
            node_ids=[messages[i]['id']],
            embedding=embeddings[i]
        )
        segments.append(seg)

    if verbose:
        typer.echo(f"Starting with {len(segments)} segments (one per message)")

    # Track merge history for elbow detection
    merge_history: List[MergeStep] = []

    # Record initial state
    merge_history.append(MergeStep(
        num_segments=len(segments),
        similarity=1.0,
        segments_snapshot=[Segment(s.start_idx, s.end_idx, s.node_ids.copy(), s.embedding) for s in segments]
    ))

    iteration = 0
    while len(segments) > min_segments:
        iteration += 1

        # Find the most similar adjacent pair
        best_similarity = -1
        best_idx = -1

        for i in range(len(segments) - 1):
            sim = compute_similarity(segments[i].embedding, segments[i + 1].embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_idx = i

        # Always stop if similarity is very low
        if best_similarity < 0.01:
            break

        # If not using elbow detection, use threshold
        if not use_elbow and best_similarity < min_similarity:
            if verbose:
                typer.echo(f"Stopping: best similarity {best_similarity:.3f} < threshold {min_similarity}")
            break

        # Merge the best pair
        seg1 = segments[best_idx]
        seg2 = segments[best_idx + 1]

        merged = Segment(
            start_idx=seg1.start_idx,
            end_idx=seg2.end_idx,
            node_ids=seg1.node_ids + seg2.node_ids,
            embedding=compute_segment_embedding(embeddings, seg1.start_idx, seg2.end_idx)
        )

        if verbose:
            typer.echo(f"Iteration {iteration}: Merging segments {best_idx} and {best_idx + 1} "
                      f"(similarity: {best_similarity:.3f}) -> {len(segments) - 1} segments")

        # Replace the two segments with the merged one
        segments = segments[:best_idx] + [merged] + segments[best_idx + 2:]

        # Record this merge step
        merge_history.append(MergeStep(
            num_segments=len(segments),
            similarity=best_similarity,
            segments_snapshot=[Segment(s.start_idx, s.end_idx, s.node_ids.copy(), s.embedding) for s in segments]
        ))

    if use_elbow and len(merge_history) > 2:
        # Find the elbow point - biggest drop in similarity
        optimal_segments = find_elbow(merge_history, verbose=verbose)
        if optimal_segments:
            segments = optimal_segments

    if verbose:
        typer.echo(f"Final: {len(segments)} segments")

    return segments


def find_elbow(merge_history: List[MergeStep], verbose: bool = False) -> Optional[List[Segment]]:
    """
    Find the elbow point in the merge history.

    The elbow is where the similarity drops most significantly,
    indicating a natural boundary between topics.

    Returns:
        Segments at the optimal stopping point, or None to use current
    """
    if len(merge_history) < 3:
        return None

    # Calculate drops in similarity between consecutive merges
    drops = []
    for i in range(1, len(merge_history)):
        prev_sim = merge_history[i - 1].similarity
        curr_sim = merge_history[i].similarity
        drop = prev_sim - curr_sim
        drops.append((i, drop, merge_history[i].num_segments))

    if verbose:
        typer.echo("\nMerge history (similarity drops):")
        for idx, drop, num_segs in drops:
            typer.echo(f"  {num_segs} segments: drop = {drop:.3f}")

    # Find the biggest drop
    if not drops:
        return None

    max_drop_idx, max_drop, _ = max(drops, key=lambda x: x[1])

    if verbose:
        typer.echo(f"\nBiggest drop: {max_drop:.3f} at step {max_drop_idx}")
        typer.echo(f"Optimal segments: {merge_history[max_drop_idx - 1].num_segments}")

    # Return the segments from BEFORE the biggest drop
    if max_drop_idx > 0:
        return merge_history[max_drop_idx - 1].segments_snapshot

    return None


def generate_topic_name(
    messages: List[Dict[str, Any]],
    segment: Segment
) -> str:
    """
    Generate a topic name for a segment.

    Uses the first user message in the segment as the basis.
    """
    from episodic.topics.topic_extraction import extract_topic_ollama

    # Build a conversation segment from messages in this segment
    segment_text = []
    for i in range(segment.start_idx, min(segment.end_idx + 1, segment.start_idx + 4)):
        msg = messages[i]
        segment_text.append(f"{msg['role']}: {msg['content'][:200]}")

    conversation_segment = "\n".join(segment_text)

    # Try LLM extraction
    try:
        name, _ = extract_topic_ollama(conversation_segment)
        if name:
            return name
    except Exception:
        pass

    # Fallback: use first user message words
    for i in range(segment.start_idx, segment.end_idx + 1):
        if messages[i]['role'] == 'user':
            words = messages[i]['content'].split()[:5]
            return '-'.join(w.lower() for w in words)[:30]

    return f"topic-{segment.start_idx}"


def reanalyze_topics(
    min_similarity: Optional[float] = None,
    min_segments: int = 1,
    apply: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Reanalyze the conversation to detect topic boundaries.

    Args:
        min_similarity: Minimum similarity threshold. If None, use elbow detection.
        min_segments: Minimum number of topics to find
        apply: If True, update the database with new topics
        verbose: Print detailed progress

    Returns:
        List of detected topics with boundaries
    """
    typer.secho("\n🔍 Reanalyzing topics...", fg=get_heading_color(), bold=True)

    # Get all messages
    messages = get_conversation_messages()
    if not messages:
        typer.secho("No messages found in conversation.", fg=get_system_color())
        return []

    typer.echo(f"Found {len(messages)} messages to analyze")

    # Initialize drift calculator for embeddings
    embedding_provider = config.get("drift_embedding_provider", "sentence-transformers")
    embedding_model = config.get("drift_embedding_model", "paraphrase-mpnet-base-v2")

    drift_calc = ConversationalDrift(
        embedding_provider=embedding_provider,
        embedding_model=embedding_model
    )

    # Compute embeddings
    typer.echo("Computing embeddings...")
    embeddings = compute_embeddings(messages, drift_calc)

    # Run hierarchical segmentation
    # If min_similarity is None, use elbow detection; otherwise use threshold
    use_elbow = min_similarity is None
    threshold = min_similarity if min_similarity is not None else 0.0

    typer.echo("Running hierarchical segmentation...")
    segments = hierarchical_segment(
        messages,
        embeddings,
        min_similarity=threshold,
        min_segments=min_segments,
        use_elbow=use_elbow,
        verbose=verbose
    )

    # Generate topic info
    topics = []
    for i, segment in enumerate(segments):
        # Find first user node in segment for start
        start_node_id = None
        for idx in range(segment.start_idx, segment.end_idx + 1):
            if messages[idx]['role'] == 'user':
                start_node_id = messages[idx]['id']
                break
        if not start_node_id:
            start_node_id = messages[segment.start_idx]['id']

        # Find last node in segment for end
        end_node_id = messages[segment.end_idx]['id']

        # Generate name
        name = generate_topic_name(messages, segment)

        topics.append({
            'name': name,
            'start_node_id': start_node_id,
            'end_node_id': end_node_id,
            'start_idx': segment.start_idx,
            'end_idx': segment.end_idx,
            'message_count': segment.end_idx - segment.start_idx + 1
        })

    # Display results
    typer.secho(f"\n📑 Detected {len(topics)} topics:", fg=get_heading_color(), bold=True)
    typer.secho("=" * 60, fg=get_heading_color())

    for i, topic in enumerate(topics):
        start_node = get_node(topic['start_node_id'])
        end_node = get_node(topic['end_node_id'])

        start_short = start_node['short_id'] if start_node else topic['start_node_id'][:8]
        end_short = end_node['short_id'] if end_node else topic['end_node_id'][:8]

        typer.secho(f"\n[{i + 1}] ", fg=get_heading_color(), bold=True, nl=False)
        typer.secho(topic['name'], fg=get_text_color(), bold=True)
        typer.echo(f"    Range: {start_short} → {end_short} ({topic['message_count']} messages)")

    if apply:
        typer.secho("\n⚠️  Applying changes to database...", fg="yellow")

        # Clear existing topics
        from episodic.db_connection import get_connection
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("DELETE FROM topics")
            conn.commit()

        # Store new topics
        for topic in topics:
            store_topic(
                name=topic['name'],
                start_node_id=topic['start_node_id'],
                end_node_id=topic['end_node_id'],
                confidence='reanalyzed'
            )

        typer.secho(f"✅ Stored {len(topics)} topics", fg="green")
    else:
        typer.secho("\n💡 Use /topics reanalyze apply to save these topics to the database", fg=get_system_color())

    return topics
