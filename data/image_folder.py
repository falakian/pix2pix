from pathlib import Path
from typing import List

# Supported image file extensions
IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
)

def is_image_file(filename: str | Path) -> bool:
    """
    Check if a file is an image based on its extension.

    Args:
        filename (str | Path): The name or path of the file to check.

    Returns:
        bool: True if the file has a supported image extension, False otherwise.
    """
    return any(str(filename).endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(dir_path: str | Path, max_dataset_size: float = float("inf")) -> List[str]:
    """
    Create a list of image file paths from a directory, up to a maximum dataset size.

    Args:
        dir_path (str | Path): Path to the directory containing images.
        max_dataset_size (float, optional): Maximum number of images to include. Defaults to infinity.

    Returns:
        List[str]: Sorted list of image file paths.

    Raises:
        ValueError: If the specified directory does not exist or is not a directory.
    """
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"'{dir_path}' is not a valid directory")

    images: List[str] = []
    for file_path in sorted(dir_path.rglob("*")):
        if file_path.is_file() and is_image_file(file_path):
            images.append(str(file_path))

    return images[:min(int(max_dataset_size), len(images))]