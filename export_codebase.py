"""
Script to export the entire codebase into a single text file.
Each file's content is wrapped with clear delimiters for easy parsing.
"""

import os

def export_codebase(source_dir, output_file, file_extensions=None):
    """
    Export all code files into a single text file with delimiters.
    
    Args:
        source_dir (str): Directory containing the codebase
        output_file (str): Path to output text file
        file_extensions (list[str], optional): List of file extensions to include.
            If None, defaults to ['.py', '.md']
    """
    if file_extensions is None:
        file_extensions = ['.py', '.md']
        
    # Ensure extensions start with dot
    file_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions]
    
    with open(output_file, 'w') as outfile:
        # Write header
        outfile.write("=== WORDLE SOLVER DQN CODEBASE ===\n")
        outfile.write(f"Exported from: {os.path.abspath(source_dir)}\n\n")
        
        # Walk through directory
        for root, _, files in os.walk(source_dir):
            for file in sorted(files):
                # Check if file has desired extension
                if any(file.endswith(ext) for ext in file_extensions):
                    # Get relative path for cleaner output
                    rel_path = os.path.relpath(os.path.join(root, file), source_dir)
                    
                    # Skip the export script itself
                    if file == os.path.basename(__file__):
                        continue
                        
                    # Skip pycache and virtual environment
                    if '__pycache__' in rel_path or 'venv' in rel_path:
                        continue
                    
                    # Write file delimiter
                    outfile.write(f"\n{'='*80}\n")
                    outfile.write(f"FILE: {rel_path}\n")
                    outfile.write(f"{'='*80}\n\n")
                    
                    # Write file contents
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as infile:
                            content = infile.read()
                            outfile.write(content)
                            # Ensure there's a newline at the end
                            if not content.endswith('\n'):
                                outfile.write('\n')
                    except Exception as e:
                        outfile.write(f"ERROR reading file: {str(e)}\n")
        
        # Write footer
        outfile.write("\n" + "="*80 + "\n")
        outfile.write("=== END OF CODEBASE ===\n")

if __name__ == "__main__":
    # Export from current directory to codebase.txt
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, "codebase.txt")
    
    export_codebase(
        source_dir=current_dir,
        output_file=output_path,
        file_extensions=['.py', '.md']  # Export both Python and Markdown files
    )
    
    print(f"Codebase exported to: {output_path}")
