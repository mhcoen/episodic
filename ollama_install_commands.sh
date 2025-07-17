#!/bin/bash
# Ollama Instruct Models Installation Script for Episodic

echo "Installing recommended instruct models for Episodic..."
echo "=================================================="

# 1. Llama 3 Instruct variants
echo "Installing Llama 3 instruct models..."
ollama pull llama3:instruct
ollama pull llama3.1:8b-instruct-q4_0  # Smaller, faster variant

# 2. Mistral Instruct (excellent for compression)
echo "Installing Mistral instruct..."
ollama pull mistral:instruct
ollama pull mistral:7b-instruct-q4_0   # Quantized version

# 3. Phi-3 (Microsoft's efficient model, great for synthesis)
echo "Installing Phi-3..."
ollama pull phi3
ollama pull phi3:mini    # Even smaller version

# 4. Optional but recommended additions
echo "Installing additional recommended models..."
ollama pull gemma:2b-instruct   # Google's small instruct model
ollama pull qwen2:1.5b          # Alibaba's efficient model

echo ""
echo "Installation complete! Installed models:"
ollama list

echo ""
echo "Recommended Episodic configuration:"
echo "===================================="
echo "/model detection ollama/llama3:instruct"
echo "/model compression ollama/mistral:instruct"
echo "/model synthesis ollama/phi3"
echo ""
echo "Or for faster/lighter setup:"
echo "/model detection ollama/phi3:mini"
echo "/model compression ollama/gemma:2b-instruct"
echo "/model synthesis ollama/qwen2:1.5b"