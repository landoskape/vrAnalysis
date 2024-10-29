import os
import datetime


def format_size(size):
    """Convert size in bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024


def analyze_directory(start_path, min_size_mb=10):
    """Analyze directory and print results immediately."""
    print(f"\nAnalyzing: {start_path}")
    print("=" * 80)

    if not os.path.exists(start_path):
        print(f"Error: Path '{start_path}' does not exist!")
        return

    # Convert min_size to bytes
    min_size = min_size_mb * 1024 * 1024

    # Dictionary to store directory sizes
    dir_sizes = {}

    # Count for progress indication
    file_count = 0

    print(f"Started at: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("Scanning directories...")

    # First pass: calculate directory sizes
    for root, dirs, files in os.walk(start_path):
        try:
            total_size = 0
            # Calculate size of files in current directory
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):  # Check if file still exists
                        total_size += os.path.getsize(file_path)
                        file_count += 1
                        if file_count % 1000 == 0:  # Progress indicator
                            print(f"Processed {file_count} files...", end="\r")
                except (PermissionError, OSError) as e:
                    continue

            dir_sizes[root] = total_size

        except (PermissionError, OSError) as e:
            print(f"\nSkipping {root}: Permission denied")
            continue

    print(f"\nCompleted scanning {file_count} files at: {datetime.datetime.now().strftime('%H:%M:%S')}")
    print("\nLargest directories:")
    print("-" * 80)

    # Calculate total sizes (including subdirectories) and sort
    total_sizes = {}
    for directory in dir_sizes:
        total_size = dir_sizes[directory]
        for subdir in dir_sizes:
            if subdir.startswith(directory) and subdir != directory:
                total_size += dir_sizes[subdir]
        if total_size >= min_size:
            total_sizes[directory] = total_size

    # Sort directories by size and print results
    sorted_dirs = sorted(total_sizes.items(), key=lambda x: x[1], reverse=True)

    if not sorted_dirs:
        print(f"No directories found larger than {min_size_mb}MB!")
        return

    # Print top directories
    for directory, size in sorted_dirs[:20]:  # Show top 20 largest directories
        try:
            # Get relative path if it's a subdirectory of start_path
            if directory.startswith(start_path):
                rel_path = os.path.relpath(directory, start_path)
                if rel_path == ".":
                    rel_path = os.path.basename(start_path)
            else:
                rel_path = directory

            # Calculate indentation based on directory depth
            depth = rel_path.count(os.sep)
            indent = "  " * depth

            print(f"{indent}{rel_path}: {format_size(size)}")

        except (OSError, ValueError) as e:
            print(f"Error processing {directory}: {e}")


if __name__ == "__main__":
    import sys

    # Get path from command line or use current directory
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = os.getcwd()
        print(f"No path specified, using current directory: {path}")

    # Get minimum size from command line or use default
    min_size = 10  # Default 10MB
    if len(sys.argv) > 2:
        try:
            min_size = float(sys.argv[2])
        except ValueError:
            print(f"Invalid minimum size specified, using default: {min_size}MB")

    analyze_directory(path, min_size)
