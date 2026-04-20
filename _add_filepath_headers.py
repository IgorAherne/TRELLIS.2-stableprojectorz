# File: add_filepath_headers.py
import os

# Configuration
# Directories to ignore to prevent corrupting environments or wasting time
IGNORE_DIRS = {
    '.git', '.idea', '.vscode', '__pycache__', 
    'venv', 'env', '.env', 'node_modules', 
    'build', 'dist', 'egg-info', 'bin', 'obj'
}

# File extensions to look for
TARGET_EXTENSIONS = ('.py', '.sh')

def add_header_to_file(file_path, relative_path):
    """Reads a file, checks for the header, and prepends it if missing."""
    
    # Create the comment string. 
    # We replace OS separator (backslash on Windows) with forward slash for consistency
    normalized_path = relative_path.replace(os.sep, '/')
    header_line = f"# {normalized_path}\n"
    
    try:
        # Read the original content
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        # Check if file is empty
        if not lines:
            existing_first_line = ""
        else:
            existing_first_line = lines[0]

        # Check if the header already exists to prevent duplication
        # We strip the newline to compare content accurately
        if existing_first_line.strip() == header_line.strip():
            print(f"[SKIPPED] Already present: {relative_path}")
            return

        # Write the new content (Header + Original content)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(header_line)
            f.writelines(lines)
            
        print(f"[UPDATED] {relative_path}")

    except Exception as e:
        print(f"[ERROR] Could not process {relative_path}: {e}")

def main():
    # Get the current working directory where the script is running
    root_dir = os.getcwd()
    
    print(f"Scanning directory: {root_dir}")
    print("------------------------------------------------")

    for current_root, dirs, files in os.walk(root_dir):
        # Modify dirs in-place to prevent os.walk from visiting ignored directories
        # This is crucial for skipping the massive 'venv' folder
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file.endswith(TARGET_EXTENSIONS):
                # Don't process this script itself if it's in the folder
                if file == os.path.basename(__file__):
                    continue

                full_path = os.path.join(current_root, file)
                
                # Calculate path relative to the project root
                rel_path = os.path.relpath(full_path, root_dir)
                
                add_header_to_file(full_path, rel_path)

    print("------------------------------------------------")
    print("Processing complete.")

if __name__ == "__main__":
    main()