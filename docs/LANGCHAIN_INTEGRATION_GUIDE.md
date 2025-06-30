# LangChain Integration Guide for Episodic

## Overview

This guide outlines how to integrate LangChain capabilities into Episodic without replacing the existing LiteLLM infrastructure. The approach uses a hybrid solution that leverages the strengths of both libraries.

## Why Hybrid Approach?

### Keep LiteLLM for:
- LLM API calls (working great with multi-provider support)
- Cost tracking (`cost_per_token` functionality)
- Unified model interface across providers
- Streaming responses
- Simple fallback handling

### Add LangChain for:
- PDF processing and document loading
- Document chunking and splitting
- Vector storage and similarity search
- RAG (Retrieval Augmented Generation)
- MCP (Model Context Protocol) integration
- Advanced tool usage patterns

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   PDF Files     │────▶│    LangChain     │────▶│  Vector Store   │
└─────────────────┘     │  (processing)    │     │   (Chroma)      │
                        └──────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐                                      │
│   MCP Server    │                                      ▼
└────────┬────────┘                              ┌─────────────────┐
         │                                       │   Enhanced      │
         ▼                                       │   Context       │
┌─────────────────┐                              └────────┬────────┘
│   MCP Client    │                                       │
└────────┬────────┘                                       ▼
         │                                       ┌─────────────────┐
         └──────────────────────────────────────▶│    LiteLLM      │
                                                 │  (LLM calls)    │
                                                 └─────────────────┘
```

## Implementation Examples

### 1. Document Manager

```python
# episodic/documents.py - NEW FILE
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Optional
import litellm

class DocumentManager:
    def __init__(self, persist_directory: Optional[str] = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.persist_directory = persist_directory
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and process a PDF file."""
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Store in vector database
        if not self.vectorstore:
            self.vectorstore = Chroma.from_documents(
                chunks, 
                self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        return chunks
    
    def query_documents(self, query: str, k: int = 4) -> List[Document]:
        """Find relevant document chunks."""
        if not self.vectorstore:
            return []
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def get_context_for_llm(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for including in LLM prompt."""
        docs = self.query_documents(query)
        
        # Truncate context if needed
        context_parts = []
        current_tokens = 0
        
        for doc in docs:
            doc_tokens = len(doc.page_content.split())
            if current_tokens + doc_tokens > max_tokens:
                break
            context_parts.append(doc.page_content)
            current_tokens += doc_tokens
        
        return "\n\n---\n\n".join(context_parts)
```

### 2. Integration with Existing Code

```python
# episodic/conversation.py - additions
from episodic.documents import DocumentManager

class ConversationManager:
    def __init__(self):
        # ... existing init code ...
        self.document_manager = DocumentManager()
        self.document_context_enabled = False
    
    def handle_chat_message_with_docs(self, user_input: str, model: str, 
                                      system_message: str, context_depth: int = 10):
        """Enhanced chat that includes document context when relevant."""
        
        # Check if we should include document context
        if self.document_context_enabled and self.document_manager.vectorstore:
            doc_context = self.document_manager.get_context_for_llm(user_input)
            
            # Enhance the user input with document context
            enhanced_input = f"""Based on the following context from loaded documents:

{doc_context}

User question: {user_input}

Please provide an answer based on the document context when relevant."""
            
            # Use existing chat message handler
            return self.handle_chat_message(
                enhanced_input, 
                model, 
                system_message, 
                context_depth
            )
        else:
            # Normal flow without documents
            return self.handle_chat_message(
                user_input, 
                model, 
                system_message, 
                context_depth
            )
```

### 3. MCP Integration

```python
# episodic/mcp_client.py - NEW FILE
import httpx
from typing import Dict, Any, List
import json

class MCPClient:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.client = httpx.Client()
    
    def list_tools(self) -> List[Dict]:
        """Get available tools from MCP server."""
        try:
            response = self.client.get(f"{self.server_url}/tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list MCP tools: {e}")
            return []
    
    def call_tool(self, tool_name: str, arguments: Dict) -> Any:
        """Execute a tool on the MCP server."""
        try:
            response = self.client.post(
                f"{self.server_url}/tools/{tool_name}",
                json=arguments
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to call MCP tool {tool_name}: {e}")
            return {"error": str(e)}

# Integration function
def query_with_mcp_tools(prompt: str, mcp_client: MCPClient, model: str = "gpt-4"):
    """Allow LLM to use MCP tools to answer questions."""
    tools = mcp_client.list_tools()
    
    if not tools:
        return query_llm(prompt, model=model)
    
    # First, ask LLM what tool to use
    tool_selection_prompt = f"""You have access to the following tools:

{json.dumps(tools, indent=2)}

User request: {prompt}

Analyze the request and respond with a JSON object containing:
- tool_name: The name of the tool to use (or "none" if no tool needed)
- arguments: The arguments to pass to the tool
- reasoning: Brief explanation of why you chose this tool
"""
    
    response, _ = query_llm(tool_selection_prompt, model=model, temperature=0)
    
    try:
        decision = json.loads(response)
        if decision.get("tool_name") != "none":
            # Call the tool
            tool_result = mcp_client.call_tool(
                decision["tool_name"], 
                decision.get("arguments", {})
            )
            
            # Generate final response with tool results
            final_prompt = f"""Tool {decision['tool_name']} returned: {tool_result}

Original request: {prompt}

Please provide a helpful response based on the tool output."""
            
            return query_llm(final_prompt, model=model)
    except Exception as e:
        logger.error(f"Error in tool selection: {e}")
    
    # Fallback to regular response
    return query_llm(prompt, model=model)
```

### 4. CLI Commands

```python
# episodic/commands/documents.py - NEW FILE
import typer
from pathlib import Path
from episodic.configuration import get_system_color

def handle_load_document(file_path: str):
    """Load a document into the conversation context."""
    path = Path(file_path)
    
    if not path.exists():
        typer.echo(f"❌ File not found: {file_path}", err=True)
        return
    
    if path.suffix.lower() != '.pdf':
        typer.echo(f"❌ Currently only PDF files are supported", err=True)
        return
    
    try:
        chunks = conversation_manager.document_manager.load_pdf(str(path))
        typer.secho(
            f"📄 Loaded {path.name} ({len(chunks)} chunks)", 
            fg=get_system_color()
        )
        
        # Show sample of content
        if chunks:
            preview = chunks[0].page_content[:200] + "..."
            typer.echo(f"Preview: {preview}")
            
    except Exception as e:
        typer.echo(f"❌ Error loading document: {e}", err=True)

def handle_docs_command(action: str = "list"):
    """Manage loaded documents."""
    if action == "list":
        # List loaded documents
        if not conversation_manager.document_manager.vectorstore:
            typer.echo("No documents loaded")
            return
            
        # Get unique sources
        sources = set()
        for doc in conversation_manager.document_manager.vectorstore.get()['documents']:
            if 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        
        typer.echo("Loaded documents:")
        for source in sources:
            typer.echo(f"  📄 {Path(source).name}")
            
    elif action == "clear":
        # Clear all documents
        conversation_manager.document_manager.vectorstore = None
        typer.secho("🗑️  Cleared all documents", fg=get_system_color())
        
    elif action == "enable":
        conversation_manager.document_context_enabled = True
        typer.secho("✅ Document context enabled", fg=get_system_color())
        
    elif action == "disable":
        conversation_manager.document_context_enabled = False
        typer.secho("❌ Document context disabled", fg=get_system_color())
```

## Installation

```bash
# Required dependencies
pip install langchain langchain-community langchain-openai pypdf chromadb tiktoken

# Optional for MCP
pip install httpx

# Optional for other document types
pip install pypdf2 pdfplumber  # Better PDF handling
pip install python-docx        # Word documents
pip install markdown           # Markdown files
```

## Configuration

Add to `episodic/config.py`:

```python
# Document processing settings
DOCUMENT_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "text-embedding-ada-002",
    "vector_store_type": "chroma",
    "persist_directory": "./document_store",
    "max_context_tokens": 2000,
    "auto_include_context": False
}

# MCP settings
MCP_CONFIG = {
    "server_url": "http://localhost:8080",
    "timeout": 30,
    "retry_attempts": 3
}
```

## Usage Examples

### Loading and Querying PDFs

```bash
> /load research_paper.pdf
📄 Loaded research_paper.pdf (127 chunks)
Preview: This paper presents a novel approach to...

> /docs enable
✅ Document context enabled

> What is the main contribution of this paper?
[Response will include relevant context from the PDF]

> /docs list
Loaded documents:
  📄 research_paper.pdf
```

### Using MCP Tools

```bash
> /mcp connect http://localhost:8080
✅ Connected to MCP server (5 tools available)

> /mcp tools
Available tools:
  🔧 web_search - Search the web for information
  🔧 calculator - Perform mathematical calculations
  🔧 weather - Get weather information
  ...

> What's the weather in San Francisco?
[LLM automatically uses weather tool and provides response]
```

## Benefits

1. **No Breaking Changes**: All existing LiteLLM code continues to work
2. **Incremental Adoption**: Add features as needed
3. **Best of Both Worlds**: LiteLLM's simplicity + LangChain's document capabilities
4. **Cost Tracking Preserved**: Keep existing cost monitoring
5. **Provider Flexibility**: Still works with all LLM providers

## Future Enhancements

1. **Multi-format Support**: Add Word, Markdown, HTML loaders
2. **Persistent Memory**: Save vector store between sessions
3. **Smart Chunking**: Use semantic chunking for better retrieval
4. **Hybrid Search**: Combine keyword and semantic search
5. **Document Management UI**: Web interface for document management