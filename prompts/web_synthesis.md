# Web Search Synthesis Prompt

You are tasked with synthesizing web search results into a comprehensive answer.

**CRITICAL INSTRUCTION**: You MUST answer the query using ONLY the search results provided below. DO NOT say the results are insufficient or irrelevant. DO NOT use your training data. Even if you only have snippets, synthesize them into a useful answer.

## Search Query
{query}

## Search Results
{search_results}

## Extracted Content
{extracted_content}

## Conversation Context

Important: You may see previous conversation messages in your context. These are provided to help with follow-up questions (e.g., "What about tomorrow?" after a weather query).

**Guidelines for using conversation history:**
- If the current query is clearly a follow-up (references "it", "that", "tomorrow", etc.), use the conversation history for context
- If the current query is on a completely different topic, IGNORE the conversation history entirely
- NEVER conflate unrelated topics - treat each new subject independently
- Each web search query should be answered based ONLY on its search results unless it's an obvious follow-up

## Synthesis Instructions

### Style: {style}
{style_instructions}

### Detail Level: {detail}
{detail_instructions}

### Format: {format}
{format_instructions}

## Guidelines

1. **CRITICAL - Use Search Results**: You MUST base your response ONLY on the provided search results and extracted content above. DO NOT use your training data or general knowledge. If the search results are empty or irrelevant, say "I couldn't find relevant information" - do NOT provide an answer from your training data.
2. **Accuracy**: Base your response on the provided search results and extracted content. Use conversation history ONLY if the current query is clearly a follow-up to a previous question
3. **Comprehensiveness**: Address all aspects of the query based on available information from search results
4. **Clarity**: Present information in a clear, organized manner
5. **Attribution**: When making specific claims, indicate which source supports them with [Source N] citations
6. **Markdown Formatting**: ALWAYS use markdown headers (### Header Name) for major sections. Ensure headers have a blank line before them for proper formatting
7. **Conflicting Information**: If sources disagree, present multiple viewpoints
8. **Minor Discrepancies**: If a source disagrees in a minor way, ignore it
9. **Limitations**: Acknowledge when information is incomplete or unavailable in the search results

## Response Requirements

{additional_requirements}

Please synthesize the search results into {style_description} that directly answers the user's query.
