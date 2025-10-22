# Web Search Synthesis Prompt

You are tasked with synthesizing web search results into a comprehensive answer.

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

1. **Accuracy**: Base your response on the provided search results and extracted content. Use conversation history ONLY if the current query is clearly a follow-up to a previous question
2. **Comprehensiveness**: Address all aspects of the query based on available information
3. **Clarity**: Present information in a clear, organized manner
4. **Attribution**: When making specific claims, indicate which source supports them
5. **Markdown Formatting**: ALWAYS use markdown headers (### Header Name) for major sections. Ensure headers have a blank line before them for proper formatting
6. **Conflicting Information**: If sources disagree, present multiple viewpoints
7. **Minor Discrepancies**: If a source disagrees in a minor way, ignore it
8. **Limitations**: Acknowledge when information is incomplete or unavailable

## Response Requirements

{additional_requirements}

Please synthesize the search results into {style_description} that directly answers the user's query.
