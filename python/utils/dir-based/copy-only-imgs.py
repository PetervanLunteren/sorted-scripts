import os
import shutil
from pathlib import Path

# conda activate "C:\Users\smart\AddaxAI_files\envs\env-base" && python "C:\Users\smart\Desktop\copy-only-imgs.py"

src_directory = r"C:\Peter\projects\2024-25-ARI\data\raw\imgs+videos+frames"
dst_directory = r"C:\Peter\projects\2024-25-ARI\data\raw\imgs+frames"

# Optional: uncomment to verify EXIF preservation
# from PIL import Image
# from PIL.ExifTags import TAGS

def copy_images_with_structure(src_dir, dst_dir, verify_exif=False):
    """
    Copy all image files from src_dir to dst_dir while preserving folder structure and EXIF data.
    
    Args:
        src_dir (str): Source directory path
        dst_dir (str): Destination directory path
        verify_exif (bool): If True, verify EXIF data preservation (requires Pillow)
    """
    # Define image extensions to look for
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    # Convert to Path objects for easier handling
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Check if source directory exists
    if not src_path.exists():
        print(f"Error: Source directory '{src_dir}' does not exist.")
        return
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    # Walk through all files and subdirectories in source
    for root, dirs, files in os.walk(src_path):
        for file in files:
            # Check if file has an image extension (case-insensitive)
            if Path(file).suffix.lower() in image_extensions:
                # Get the source file path
                src_file = Path(root) / file
                
                # Calculate relative path from source directory
                rel_path = src_file.relative_to(src_path)
                
                # Create destination file path
                dst_file = dst_path / rel_path
                
                # Create destination directory if it doesn't exist
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Copy the file with all metadata (including EXIF)
                    shutil.copy2(str(src_file), str(dst_file))
                    print(f"Copied: {rel_path}")
                    
                    # Optional EXIF verification
                    if verify_exif and src_file.suffix.lower() in {'.jpg', '.jpeg'}:
                        verify_exif_preservation(src_file, dst_file)
                    
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying {rel_path}: {e}")
    
    print(f"\nCompleted! Copied {copied_count} image files with all EXIF data preserved.")
    
    # No need to remove empty directories since we're copying
    print("Copy operation completed successfully!")

def verify_exif_preservation(src_file, dst_file):
    """
    Verify that EXIF data was preserved during copy (requires Pillow).
    
    Args:
        src_file (Path): Source file path
        dst_file (Path): Destination file path
    """
    try:
        from PIL import Image
        
        # Read EXIF from source
        with Image.open(src_file) as src_img:
            src_exif = src_img._getexif()
        
        # Read EXIF from destination
        with Image.open(dst_file) as dst_img:
            dst_exif = dst_img._getexif()
        
        if src_exif == dst_exif:
            print(f"  ✓ EXIF data preserved")
        else:
            print(f"  ⚠ EXIF data may have changed")
            
    except ImportError:
        print("  Note: Install Pillow (pip install Pillow) to verify EXIF preservation")
    except Exception as e:
        print(f"  Note: Could not verify EXIF data: {e}")

def remove_empty_dirs(path):
    """
    Remove empty directories recursively (not needed for copy operations).
    Kept for reference only.
    
    Args:
        path (Path): Directory path to clean up
    """
    pass  # Not needed for copying

if __name__ == "__main__":
    # Example usage

    
    # Uncomment and modify the paths below to use
    copy_images_with_structure(src_directory, dst_directory, verify_exif=False)
    
    print("Script ready! This will copy all images with EXIF data preserved.")
    print("Set verify_exif=True to check EXIF preservation (requires: pip install Pillow)")