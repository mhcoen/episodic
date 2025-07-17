"""
Asynchronous background compression system for episodic conversations.

This module provides automatic compression of conversation segments based on
topic boundaries, running in background threads to avoid blocking the main
conversation flow.
"""

import threading
import queue
from typing import Dict, Any, List
from datetime import datetime

from episodic.db import (
    get_ancestry
)
from episodic.llm import query_llm
from episodic.config import config
import typer


class CompressionJob:
    """Represents a compression job in the queue."""
    
    def __init__(self, start_node_id: str, end_node_id: str, 
                 topic_name: str, priority: int = 5):
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.topic_name = topic_name
        self.priority = priority
        self.created_at = datetime.now()
        self.attempts = 0
        
    def __lt__(self, other):
        """Priority queue comparison - lower priority number = higher priority."""
        return self.priority < other.priority


class AsyncCompressionManager:
    """Manages background compression of conversation segments."""
    
    def __init__(self):
        self.compression_queue = queue.PriorityQueue()
        self.worker_thread = None
        self.shutdown_event = threading.Event()
        self.compression_lock = threading.Lock()
        self.stats = {
            'total_compressed': 0,
            'failed_compressions': 0,
            'total_words_saved': 0,
            'queue_size': 0
        }
        
    def start(self):
        """Start the background compression worker."""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(
                target=self._compression_worker,
                name="CompressionWorker",
                daemon=True
            )
            self.worker_thread.start()
            if config.get('debug'):
                typer.echo("🔄 Background compression worker started")
    
    def stop(self):
        """Stop the background compression worker gracefully."""
        if self.worker_thread and self.worker_thread.is_alive():
            self.shutdown_event.set()
            # Add sentinel to wake up the worker
            self.compression_queue.put((float('inf'), None))
            self.worker_thread.join(timeout=5.0)
            if config.get('debug'):
                typer.echo("🛑 Background compression worker stopped")
    
    def queue_compression(self, start_node_id: str, end_node_id: str, 
                         topic_name: str, priority: int = 5):
        """Queue a topic segment for compression."""
        job = CompressionJob(start_node_id, end_node_id, topic_name, priority)
        self.compression_queue.put((job.priority, job))
        self.stats['queue_size'] = self.compression_queue.qsize()
        
        if config.get('debug'):
            typer.echo(f"📥 Queued compression job for topic '{topic_name}'")
    
    def _compression_worker(self):
        """Background worker that processes compression jobs."""
        if config.get('debug'):
            typer.echo(f"🔧 Compression worker started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get job with timeout to check shutdown periodically
                priority, job = self.compression_queue.get(timeout=1.0)
                
                if job is None:  # Sentinel value for shutdown
                    break
                
                if config.get('debug'):
                    typer.echo(f"🔧 Processing compression job for topic '{job.topic_name}'")
                
                # Process the compression job
                success = self._compress_topic_segment(job)
                
                if not success and job.attempts < 3:
                    # Retry failed jobs with lower priority
                    job.attempts += 1
                    job.priority += 2  # Lower priority for retry
                    self.compression_queue.put((job.priority, job))
                    if config.get('debug'):
                        typer.echo(f"🔄 Retrying compression for topic '{job.topic_name}' (attempt {job.attempts})")
                elif not success:
                    self.stats['failed_compressions'] += 1
                    if config.get('debug'):
                        typer.echo(f"❌ Failed to compress topic '{job.topic_name}' after 3 attempts")
                
                self.stats['queue_size'] = self.compression_queue.qsize()
                
            except queue.Empty:
                continue
            except Exception as e:
                if config.get('debug'):
                    typer.echo(f"⚠️  Compression worker error: {e}")
    
    def _compress_topic_segment(self, job: CompressionJob) -> bool:
        """
        Compress a topic segment into a summary node.
        
        Returns:
            True if compression succeeded, False otherwise
        """
        try:
            if config.get('debug'):
                typer.echo(f"🔧 Compressing topic '{job.topic_name}' from {job.start_node_id} to {job.end_node_id}")
            
            # Get nodes in the topic segment
            nodes = self._get_topic_nodes(job.start_node_id, job.end_node_id)
            
            if config.get('debug'):
                typer.echo(f"🔧 Found {len(nodes)} nodes in topic segment")
            
            if len(nodes) < config.get('compression_min_nodes', 5):
                # Skip compression for very short topics
                if config.get('debug'):
                    typer.echo(f"🔧 Skipping compression - only {len(nodes)} nodes (min: {config.get('compression_min_nodes', 5)})")
                return True
            
            # Build conversation text
            conversation_text = self._format_nodes_for_compression(nodes)
            
            # Load compression prompt template if available
            from pathlib import Path
            prompt_path = Path(__file__).parent.parent / 'prompts' / 'compression.md'
            
            if prompt_path.exists():
                # Use template-based prompt
                prompt_template = prompt_path.read_text()
                prompt = prompt_template.format(
                    topic_name=job.topic_name,
                    conversation_text=conversation_text
                )
            else:
                # Fallback to simple prompt
                prompt = f"""Compress this conversation about '{job.topic_name}' into a concise summary.
Focus on key insights, decisions, conclusions, and important details.
Preserve the essential information while reducing redundancy.

Conversation ({len(nodes)} messages):
{conversation_text}

Concise summary:"""
            
            # Use fast model for background compression with compression parameters
            compression_model = config.get('compression_model', 'ollama/llama3')
            compression_params = config.get_model_params('compression', model=compression_model)
            summary, metadata = query_llm(prompt, model=compression_model, **compression_params)
            
            if not summary:
                return False
            
            # Prepare compressed content
            compressed_content = f"[Compressed Topic: {job.topic_name}]\n\n{summary}"
            
            # Calculate compression metrics
            original_words = sum(len(node.get('content', '').split()) for node in nodes)
            compressed_words = len(summary.split())
            compression_ratio = (1 - compressed_words / original_words) * 100 if original_words > 0 else 0
            
            # Get all node IDs in the compressed range
            node_ids = [node['id'] for node in nodes]
            
            # Store compression without inserting into conversation tree
            compression_id = store_compression_v2(
                content=compressed_content,
                start_node_id=job.start_node_id,
                end_node_id=job.end_node_id,
                node_ids=node_ids,
                original_node_count=len(nodes),
                original_words=original_words,
                compressed_words=compressed_words,
                compression_ratio=compression_ratio,
                strategy='auto-topic',
                duration_seconds=None  # Not tracking for background jobs
            )
            
            # Update stats
            with self.compression_lock:
                self.stats['total_compressed'] += 1
                self.stats['total_words_saved'] += (original_words - compressed_words)
            
            if config.get('show_compression_notifications', False):
                typer.echo(f"\n✅ Auto-compressed topic '{job.topic_name}' ({compression_ratio:.1f}% reduction)")
            
            return True
            
        except Exception as e:
            if config.get('debug'):
                typer.echo(f"⚠️  Compression error for topic '{job.topic_name}': {e}")
            return False
    
    def _get_topic_nodes(self, start_node_id: str, end_node_id: str) -> List[Dict]:
        """Get all nodes between start and end of a topic."""
        try:
            # This is a simplified version - in production would need proper DAG traversal
            end_ancestry = get_ancestry(end_node_id)
            
            if not end_ancestry:
                if config.get('debug'):
                    typer.echo(f"⚠️  No ancestry found for end node: {end_node_id}")
                return []
            
            # Find nodes between start and end
            nodes = []
            collecting = False
            for node in end_ancestry:
                if node['id'] == start_node_id:
                    collecting = True
                if collecting:
                    nodes.append(node)
                if node['id'] == end_node_id:
                    break
            
            if config.get('debug') and not collecting:
                typer.echo(f"⚠️  Start node {start_node_id} not found in ancestry of {end_node_id}")
            
            return nodes
        except Exception as e:
            if config.get('debug'):
                typer.echo(f"⚠️  Error getting topic nodes: {e}")
            return []
    
    def _format_nodes_for_compression(self, nodes: List[Dict]) -> str:
        """Format nodes for compression prompt."""
        lines = []
        for node in nodes:
            role = "You" if node.get('role') == 'assistant' else "User"
            content = node.get('content', '').strip()
            if content:
                lines.append(f"{role}: {content}")
        return "\n\n".join(lines)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression manager statistics."""
        with self.compression_lock:
            return self.stats.copy()
    
    def get_queue_info(self) -> List[Dict[str, Any]]:
        """Get information about pending compression jobs."""
        # Note: This is a snapshot and may not reflect real-time state
        pending = []
        temp_items = []
        
        # Drain queue temporarily to inspect
        while not self.compression_queue.empty():
            try:
                item = self.compression_queue.get_nowait()
                temp_items.append(item)
                priority, job = item
                if job:
                    pending.append({
                        'topic': job.topic_name,
                        'priority': job.priority,
                        'created': job.created_at.isoformat(),
                        'attempts': job.attempts
                    })
            except queue.Empty:
                break
        
        # Put items back
        for item in temp_items:
            self.compression_queue.put(item)
        
        return pending


# Global compression manager instance
compression_manager = AsyncCompressionManager()


def start_auto_compression():
    """Start the automatic compression system."""
    # Ensure compression tables exist
    # Tables are created during database initialization now
    compression_manager.start()


def stop_auto_compression():
    """Stop the automatic compression system."""
    compression_manager.stop()


def queue_topic_for_compression(start_node_id: str, end_node_id: str, 
                               topic_name: str, priority: int = 5):
    """Queue a topic segment for background compression."""
    if config.get('auto_compress_topics', True):
        compression_manager.queue_compression(
            start_node_id, end_node_id, topic_name, priority
        )