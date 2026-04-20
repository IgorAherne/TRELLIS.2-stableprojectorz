# mark_files.py
import os

def add_filepath_markers():
    # Get the directory where this script is located
    root_dir = os.path.dirname(os.path.abspath(__file__))
    this_script_name = os.path.basename(__file__)

    # Directories to exclude from the search
    # (Based on your provided directory list to avoid corrupting environments)
    EXCLUDE_DIRS = {
        '.git', '.vs', 'venv', 'whl', 
        'trellis2.egg-info', '__pycache__', 
        'build', 'tmp', 'assets'
    }

    print(f"--- Starting Search in: {root_dir} ---")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to prevent walking into excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for filename in filenames:
            if filename.endswith('.py'):
                # Skip this marker script itself
                if filename == this_script_name and dirpath == root_dir:
                    continue

                full_path = os.path.join(dirpath, filename)
                
                # Calculate relative path
                rel_path = os.path.relpath(full_path, root_dir)
                
                # Normalize slashes to forward slashes for the comment (standard in code)
                # If you prefer backslashes, remove the .replace part
                display_path = rel_path.replace(os.sep, '/')
                
                header_line = f"# File: {display_path}\n"

                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    print(f"[SKIP] Encoding error (not UTF-8): {rel_path}")
                    continue
                except Exception as e:
                    print(f"[SKIP] Error reading {rel_path}: {e}")
                    continue

                # Check if file is already marked
                # We check the first few lines to see if our header exists
                already_marked = False
                for i in range(min(3, len(lines))):
                    if lines[i].strip() == header_line.strip():
                        already_marked = True
                        break
                
                if already_marked:
                    print(f"[SKIP] Already marked: {rel_path}")
                    continue

                # Determine insertion point (Handle Shebangs and Encoding cookies)
                # Python scripts break if markers are placed before #! or coding definitions
                insert_idx = 0
                if lines:
                    if lines[0].startswith("#!"):
                        insert_idx += 1
                        # If shebang exists, encoding cookie might be on line 2
                        if len(lines) > 1 and ("coding:" in lines[1] or "coding=" in lines[1]):
                            insert_idx += 1
                    elif "coding:" in lines[0] or "coding=" in lines[0]:
                        insert_idx += 1

                # Insert the header
                lines.insert(insert_idx, header_line)

                # Write back to file
                try:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.writelines(lines)
                    print(f"[DONE] Marked: {rel_path}")
                except Exception as e:
                    print(f"[FAIL] Could not write to {rel_path}: {e}")

if __name__ == "__main__":
    add_filepath_markers()
    print("--- Processing Complete ---")