import os

def find_files_with_extension(directory, extensions):
    """
    Find all files in a directory with given extensions.

    :param directory: Path to the directory to search.
    :param extensions: A single file extension or a list of file extensions.
    :return: List of paths to files with matching extensions.
    """
    if isinstance(extensions, str):
        extensions = [extensions]

    matching_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            for ext in extensions:
                if file.endswith(ext):
                    matching_files.append(os.path.join(root, file))
                    break

    return matching_files