"""
Script to export the entire codebase into a markdown file.
Each file's content is wrapped with markdown code blocks and headers.
"""

import os

def export_codebase(source_dir, output_file, file_extensions=None):
    """
    Export all code files into a single markdown file with proper formatting.
    
    Args:
        source_dir (str): Directory containing the codebase
        output_file (str): Path to output markdown file
        file_extensions (list[str], optional): List of file extensions to include.
            If None, defaults to ['.py', '.md']
    """
    if file_extensions is None:
        file_extensions = ['.py', '.md']
        
    # Ensure extensions start with dot
    file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    
    # Define core files in desired order
    core_files = [
        'main.py',
        'agent.py',
        'environment.py',
        'train.py',
        'play_games.py',
        'utils.py',
        'replay_buffer.py'
    ]
    
    with open(output_file, 'w') as outfile:
        # Write header
        outfile.write("# Wordle Solver DQN Codebase\n\n")
        outfile.write("This document contains the complete codebase for the Wordle Solver DQN project.\n\n")
        
        # Write table of contents
        outfile.write("## Table of Contents\n\n")
        outfile.write("### Core Files\n")
        for filename in core_files:
            clean_name = os.path.splitext(filename)[0]
            outfile.write(f"- [{clean_name}](#{clean_name})\n")
        
        outfile.write("\n### Test Files\n")
        test_dir = os.path.join(source_dir, 'tests')
        if os.path.exists(test_dir):
            for file in sorted(os.listdir(test_dir)):
                if file.startswith('test_') and file.endswith('.py'):
                    clean_name = os.path.splitext(file)[0]
                    outfile.write(f"- [{clean_name}](#{clean_name})\n")
        
        outfile.write("\n---\n\n")
        
        # Write core files first
        outfile.write("## Core Files\n\n")
        for filename in core_files:
            file_path = os.path.join(source_dir, filename)
            if os.path.exists(file_path):
                clean_name = os.path.splitext(filename)[0]
                outfile.write(f"### {clean_name}\n\n")
                outfile.write("```python\n")
                with open(file_path, 'r') as infile:
                    outfile.write(infile.read())
                outfile.write("\n```\n\n")
        
        # Write test files
        outfile.write("## Test Files\n\n")
        if os.path.exists(test_dir):
            for file in sorted(os.listdir(test_dir)):
                if file.startswith('test_') and file.endswith('.py'):
                    file_path = os.path.join(test_dir, file)
                    clean_name = os.path.splitext(file)[0]
                    outfile.write(f"### {clean_name}\n\n")
                    outfile.write("```python\n")
                    with open(file_path, 'r') as infile:
                        outfile.write(infile.read())
                    outfile.write("\n```\n\n")

if __name__ == "__main__":
    # Export from project root to CODEBASE.md
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)  # Go up one level to project root
    output_path = os.path.join(project_root, "CODEBASE.md")
    
    print(f"Exporting codebase from {project_root}")
    export_codebase(project_root, output_path)
    print(f"Codebase exported to {output_path}")
