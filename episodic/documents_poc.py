"""
Proof of Concept for PDF document loading and querying in Episodic.
This demonstrates how to add document context without disrupting existing functionality.
"""

from typing import List, Optional, Tuple
import logging
from pathlib import Path

# Use conditional imports to avoid breaking existing code
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not installed. Document features will be disabled.")

from episodic.llm import query_llm
from episodic.config import config

logger = logging.getLogger(__name__)


class DocumentManagerPOC:
    """Proof of concept document manager for Episodic."""
    
    def __init__(self):
        if not LANGCHAIN_AVAILABLE:
            self.enabled = False
            return
            
        self.enabled = True
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.loaded_documents = {}  # filename -> doc chunks
        
        # Configure text splitter for optimal chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,      # Smaller chunks for better precision
            chunk_overlap=50,    # Some overlap to maintain context
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> Tuple[bool, str]:
        """
        Load a PDF file and add it to the vector store.
        
        Returns:
            Tuple of (success, message)
        """
        if not self.enabled:
            return False, "Document features not available (LangChain not installed)"
        
        try:
            path = Path(pdf_path)
            if not path.exists():
                return False, f"File not found: {pdf_path}"
            
            if path.suffix.lower() != '.pdf':
                return False, "Only PDF files are supported in this POC"
            
            # Load PDF
            loader = PyPDFLoader(str(path))
            documents = loader.load()
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_index'] = i
                chunk.metadata['source_name'] = path.name
            
            # Store in vector database
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(
                    chunks, 
                    self.embeddings,
                    collection_name="episodic_docs"
                )
            else:
                self.vectorstore.add_documents(chunks)
            
            # Track loaded documents
            self.loaded_documents[path.name] = chunks
            
            return True, f"Successfully loaded {path.name} ({len(chunks)} chunks, ~{len(documents)} pages)"
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return False, f"Error loading PDF: {str(e)}"
    
    def find_relevant_context(self, query: str, k: int = 3, score_threshold: float = 0.7) -> List[str]:
        """Find the most relevant document chunks for a query."""
        if not self.enabled or not self.vectorstore:
            return []
        
        try:
            # Search for relevant documents with scores
            relevant_docs = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=k
            )
            
            # Filter by relevance score
            contexts = []
            for doc, score in relevant_docs:
                # Only include if score meets threshold
                if score >= score_threshold:
                    source = doc.metadata.get('source_name', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    
                    context = f"[From {source}, page {page}]:\n{doc.page_content}"
                    contexts.append(context)
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def enhance_prompt_with_context(self, user_prompt: str, score_threshold: float = 0.7) -> str:
        """Enhance a user prompt with relevant document context."""
        if not self.enabled or not self.vectorstore:
            return user_prompt
        
        # Find relevant context with threshold
        contexts = self.find_relevant_context(user_prompt, score_threshold=score_threshold)
        
        if not contexts:
            return user_prompt
        
        # Log what we found (for debugging)
        logger.info(f"Found {len(contexts)} relevant contexts for query: {user_prompt[:50]}...")
        
        # Build enhanced prompt
        context_section = "\n\n".join(contexts)
        
        enhanced_prompt = f"""Context from project documentation:

{context_section}

User question: {user_prompt}

Please answer the user's question. Use the documentation context ONLY if it's directly relevant to answering the question. For general questions unrelated to the documentation, just provide a direct answer without referring to the context."""
        
        return enhanced_prompt
    
    def get_status(self) -> str:
        """Get current status of document manager."""
        if not self.enabled:
            return "Document features disabled (LangChain not installed)"
        
        if not self.loaded_documents:
            return "No documents loaded"
        
        doc_list = "\n".join([f"  - {name} ({len(chunks)} chunks)" 
                             for name, chunks in self.loaded_documents.items()])
        
        return f"Loaded documents:\n{doc_list}"


# Demonstration functions
def demo_basic_usage():
    """Demonstrate basic PDF loading and querying."""
    print("=== PDF Loading POC Demo ===\n")
    
    # Initialize document manager
    doc_manager = DocumentManagerPOC()
    
    # Example: Load a PDF
    success, message = doc_manager.load_pdf("example.pdf")
    print(f"Load result: {message}\n")
    
    if success:
        # Example: Query without context
        basic_response, _ = query_llm(
            "What is machine learning?",
            model=config.get("model", "gpt-4o-mini")
        )
        print("Response WITHOUT document context:")
        print(basic_response[:200] + "...\n")
        
        # Example: Query with context
        enhanced_prompt = doc_manager.enhance_prompt_with_context(
            "What is machine learning?"
        )
        enhanced_response, _ = query_llm(
            enhanced_prompt,
            model=config.get("model", "gpt-4o-mini")
        )
        print("Response WITH document context:")
        print(enhanced_response[:200] + "...\n")
    
    # Show status
    print("Document manager status:")
    print(doc_manager.get_status())


def demo_integration_example(conversation_manager):
    """Show how this would integrate with existing ConversationManager."""
    
    # Add document manager to conversation manager
    if not hasattr(conversation_manager, 'doc_manager'):
        conversation_manager.doc_manager = DocumentManagerPOC()
    
    # Original handle_chat_message method would be wrapped
    original_handle = conversation_manager.handle_chat_message
    
    def handle_with_docs(user_input: str, model: str, system_message: str, 
                        context_depth: int = 10):
        # Check if we should enhance with document context
        if conversation_manager.doc_manager.enabled and \
           conversation_manager.doc_manager.vectorstore:
            # Enhance the prompt
            enhanced_input = conversation_manager.doc_manager.enhance_prompt_with_context(
                user_input
            )
            # Use enhanced input if documents were found
            if enhanced_input != user_input:
                print("[System: Including relevant document context]")
                user_input = enhanced_input
        
        # Call original method
        return original_handle(user_input, model, system_message, context_depth)
    
    # Replace method
    conversation_manager.handle_chat_message = handle_with_docs
    
    return conversation_manager


if __name__ == "__main__":
    # Run the demo
    demo_basic_usage()