#!/usr/bin/env python3
"""
Script to compile all documentation files in the Episodic project into a single PDF.
"""

import os
import markdown2
from weasyprint import HTML, CSS
from datetime import datetime

# Define the order and structure of documentation
DOCUMENTATION_STRUCTURE = [
    {
        'title': 'Main Documentation',
        'files': [
            ('README.md', 'Episodic - Main README'),
            ('USAGE_GUIDE.md', 'Usage Guide'),
            ('CLAUDE.md', 'Claude Integration Guide'),
        ]
    },
    {
        'title': 'Installation and Setup',
        'files': [
            ('docs/Installation.md', 'Installation Guide'),
            ('docs/CLIReference.md', 'CLI Reference'),
            ('docs/LLMProviders.md', 'LLM Providers'),
        ]
    },
    {
        'title': 'Core Features',
        'files': [
            ('docs/AdvancedUsage.md', 'Advanced Usage'),
            ('docs/Visualization.md', 'Visualization'),
            ('docs/structure.md', 'Project Structure'),
        ]
    },
    {
        'title': 'Development',
        'files': [
            ('docs/Development.md', 'Development Guide'),
            ('TODO.md', 'TODO List'),
            ('ImplementationPlan.md', 'Implementation Plan'),
        ]
    },
    {
        'title': 'Technical Documentation',
        'files': [
            ('COMPRESSION_AND_TOPIC_FIXES.md', 'Compression and Topic Fixes'),
            ('TOPIC_DETECTION_FIXES.md', 'Topic Detection Fixes'),
            ('topic_detection_fix.md', 'Topic Detection Implementation'),
            ('STREAMING_IMPLEMENTATION.md', 'Streaming Implementation'),
            ('STREAMING_FIX_SUMMARY.md', 'Streaming Fix Summary'),
            ('PromptCaching.md', 'Prompt Caching'),
            ('AsyncCompressionDesign.md', 'Async Compression Design'),
            ('ConversationalDriftDesign.md', 'Conversational Drift Design'),
        ]
    },
    {
        'title': 'Integration Guides',
        'files': [
            ('docs/LANGCHAIN_INTEGRATION_GUIDE.md', 'LangChain Integration Guide'),
            ('docs/HYBRID_TOPIC_DETECTION_DESIGN.md', 'Hybrid Topic Detection Design'),
        ]
    },
    {
        'title': 'Testing Documentation',
        'files': [
            ('tests/README.md', 'Tests README'),
            ('tests/ORGANIZED_TESTS.md', 'Organized Tests Guide'),
            ('scripts/README.md', 'Scripts README'),
            ('scripts/topic/README.md', 'Topic Testing README'),
        ]
    }
]

# CSS for PDF styling
PDF_CSS = """
@page {
    size: A4;
    margin: 2cm;
    @top-center {
        content: "Episodic Documentation";
        font-size: 10pt;
        color: #666;
    }
    @bottom-right {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 10pt;
        color: #666;
    }
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
    margin-top: 40px;
    margin-bottom: 20px;
    page-break-before: always;
}

h1:first-of-type {
    page-break-before: avoid;
}

h2 {
    color: #34495e;
    margin-top: 30px;
    margin-bottom: 15px;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 5px;
}

h3 {
    color: #7f8c8d;
    margin-top: 20px;
    margin-bottom: 10px;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
    font-family: "Courier New", Courier, monospace;
    font-size: 0.9em;
}

pre {
    background-color: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 15px;
    overflow-x: auto;
    line-height: 1.4;
    page-break-inside: avoid;
}

pre code {
    background-color: transparent;
    padding: 0;
}

blockquote {
    border-left: 4px solid #3498db;
    padding-left: 15px;
    margin-left: 0;
    color: #666;
    font-style: italic;
}

ul, ol {
    margin-bottom: 15px;
}

li {
    margin-bottom: 5px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #ddd;
    padding: 12px;
    text-align: left;
}

th {
    background-color: #f8f9fa;
    font-weight: bold;
    color: #2c3e50;
}

tr:nth-child(even) {
    background-color: #f8f9fa;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

.toc {
    background-color: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 20px;
    margin-bottom: 40px;
}

.toc h2 {
    margin-top: 0;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
}

.toc ul {
    list-style-type: none;
    padding-left: 0;
}

.toc li {
    margin-bottom: 8px;
}

.toc a {
    color: #34495e;
    font-weight: 500;
}

.section-divider {
    margin: 60px 0;
    text-align: center;
    color: #bdc3c7;
}

.timestamp {
    text-align: right;
    color: #7f8c8d;
    font-size: 0.9em;
    margin-bottom: 20px;
}
"""

def read_file(filepath):
    """Read a markdown file and return its content."""
    full_path = os.path.join('/Users/mhcoen/proj/episodic', filepath)
    if os.path.exists(full_path):
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    return None

def create_table_of_contents():
    """Create a table of contents in HTML."""
    toc_html = '<div class="toc">\n<h2>Table of Contents</h2>\n<ul>\n'
    
    for section in DOCUMENTATION_STRUCTURE:
        toc_html += f'<li><strong>{section["title"]}</strong>\n<ul>\n'
        for filepath, title in section['files']:
            anchor = title.lower().replace(' ', '-').replace('/', '-')
            toc_html += f'<li><a href="#{anchor}">{title}</a></li>\n'
        toc_html += '</ul></li>\n'
    
    toc_html += '</ul>\n</div>\n'
    return toc_html

def compile_documentation():
    """Compile all documentation into a single HTML string."""
    html_parts = []
    
    # Add title page
    html_parts.append(f"""
    <h1 style="text-align: center; margin-top: 100px; font-size: 48px; border-bottom: none;">
        Episodic Documentation
    </h1>
    <p style="text-align: center; font-size: 24px; color: #7f8c8d; margin-top: 20px;">
        Conversational DAG-based Memory Agent
    </p>
    <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """)
    
    # Add table of contents
    html_parts.append(create_table_of_contents())
    
    # Process each section
    for section in DOCUMENTATION_STRUCTURE:
        # Add section header
        html_parts.append(f'<div class="section-divider">• • •</div>')
        
        for filepath, title in section['files']:
            content = read_file(filepath)
            if content:
                # Create anchor for TOC
                anchor = title.lower().replace(' ', '-').replace('/', '-')
                
                # Convert markdown to HTML
                html_content = markdown2.markdown(
                    content,
                    extras=['fenced-code-blocks', 'tables', 'header-ids', 'strike', 'task_list']
                )
                
                # Add the document with title
                html_parts.append(f'<h1 id="{anchor}">{title}</h1>')
                html_parts.append(f'<p style="color: #7f8c8d; font-size: 0.9em;">Source: {filepath}</p>')
                html_parts.append(html_content)
            else:
                print(f"Warning: Could not find {filepath}")
    
    # Combine all HTML parts
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Episodic Documentation</title>
    </head>
    <body>
        {''.join(html_parts)}
    </body>
    </html>
    """
    
    return full_html

def main():
    """Main function to generate the PDF."""
    print("Compiling Episodic documentation...")
    
    # Compile all documentation
    html_content = compile_documentation()
    
    # Create PDF
    print("Generating PDF...")
    html = HTML(string=html_content)
    css = CSS(string=PDF_CSS)
    
    output_path = '/Users/mhcoen/proj/episodic/example.pdf'
    html.write_pdf(output_path, stylesheets=[css])
    
    print(f"PDF successfully generated: {output_path}")
    
    # Get file size
    file_size = os.path.getsize(output_path)
    print(f"File size: {file_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()