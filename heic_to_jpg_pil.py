#!/usr/bin/env python3
"""
HEIC to JPG Converter (Python Libraries Version)
Converts all HEIC files to JPG format using Python libraries (pillow-heif)
"""

import os
import argparse
from pathlib import Path
import sys

def check_dependencies():
    """Check if required Python libraries are installed"""
    try:
        from PIL import Image
        print("✅ Pillow (PIL) found")
    except ImportError:
        print("❌ Pillow not found. Install with: pip install Pillow")
        return False
    
    try:
        from pillow_heif import register_heif_opener
        print("✅ pillow-heif found")
        return True
    except ImportError:
        print("❌ pillow-heif not found. Install with:")
        print("   pip install pillow-heif")
        print("   or")
        print("   pip install pillow-heif[pyheif]")
        return False

def convert_heic_to_jpg_pil(heic_path, jpg_path, quality=90):
    """Convert a single HEIC file to JPG using Pillow"""
    try:
        from PIL import Image
        from pillow_heif import register_heif_opener
        
        # Register HEIF opener with Pillow
        register_heif_opener()
        
        # Open and convert
        with Image.open(heic_path) as img:
            # Convert to RGB (remove alpha channel if present)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPG
            img.save(jpg_path, 'JPEG', quality=quality, optimize=True)
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def convert_heic_files_pil(root_dir, quality=90, remove_original=True, dry_run=False):
    """
    Convert all HEIC files to JPG using Python libraries
    
    Args:
        root_dir (str): Root directory to search for HEIC files
        quality (int): JPG quality (1-100, default 90)
        remove_original (bool): Whether to remove original HEIC files after conversion (default: True)
        dry_run (bool): If True, only show what would be converted without actually converting
    """
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ Directory does not exist: {root_dir}")
        return False
    
    if not root_path.is_dir():
        print(f"❌ Path is not a directory: {root_dir}")
        return False
    
    print(f"📂 Searching for HEIC files in: {root_dir}")
    print(f"🎨 JPG quality: {quality}")
    print(f"🗑️  Remove originals: {remove_original}")
    print(f"🔄 Dry run: {dry_run}")
    print("-" * 60)
    
    # Find all HEIC files recursively
    heic_extensions = ['.heic', '.HEIC', '.heif', '.HEIF']
    heic_files = []
    
    for ext in heic_extensions:
        heic_files.extend(root_path.rglob(f"*{ext}"))
    
    if not heic_files:
        print("❌ No HEIC/HEIF files found in the directory tree")
        return False
    
    print(f"📱 Found {len(heic_files)} HEIC/HEIF files to convert")
    print()
    
    converted_count = 0
    failed_count = 0
    skipped_count = 0
    
    for heic_file in heic_files:
        # Generate JPG filename
        jpg_file = heic_file.with_suffix('.jpg')
        
        # Get relative path for display
        try:
            rel_heic = heic_file.relative_to(root_path)
            rel_jpg = jpg_file.relative_to(root_path)
        except ValueError:
            rel_heic = heic_file
            rel_jpg = jpg_file
        
        print(f"🔄 Processing: {rel_heic}")
        
        # Check if JPG already exists
        if jpg_file.exists():
            print(f"  ⚠️  JPG already exists: {rel_jpg}")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"  📋 Would convert: {rel_heic} → {rel_jpg}")
            if remove_original:
                print(f"  📋 Would remove: {rel_heic}")
            converted_count += 1
            continue
        
        # Convert the file
        success, error = convert_heic_to_jpg_pil(heic_file, jpg_file, quality)
        
        if success:
            print(f"  ✅ Converted: {rel_jpg}")
            converted_count += 1
            
            # Remove original if requested
            if remove_original:
                try:
                    heic_file.unlink()
                    print(f"  🗑️  Removed original: {rel_heic}")
                except Exception as e:
                    print(f"  ⚠️  Could not remove original: {e}")
        else:
            print(f"  ❌ Failed to convert: {error}")
            failed_count += 1
        
        print()
    
    print("=" * 60)
    print(f"📊 Conversion Summary:")
    print(f"  ✅ Successfully converted: {converted_count} files")
    print(f"  ❌ Failed conversions: {failed_count} files")
    print(f"  ⚠️  Skipped (already exist): {skipped_count} files")
    print(f"  📱 Total HEIC files found: {len(heic_files)}")
    
    return converted_count > 0 or dry_run

def main():
    parser = argparse.ArgumentParser(description="Convert all HEIC files to JPG using Python libraries (deletes originals by default)")
    parser.add_argument("directory", help="Root directory to search for HEIC files")
    parser.add_argument("--quality", "-q", type=int, default=90, choices=range(1, 101),
                       help="JPG quality (1-100, default: 90)")
    parser.add_argument("--keep-original", "-k", action="store_true",
                       help="Keep original HEIC files after conversion (default: delete them)")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Show what would be converted without actually converting")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    success = convert_heic_files_pil(
        args.directory,
        args.quality,
        not args.keep_original,  # remove_original = True unless --keep-original is specified
        args.dry_run
    )
    
    if success:
        print("\n🎉 Operation completed successfully!")
    else:
        print("\n💥 Operation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) == 1:
        print("📱 HEIC to JPG Converter (Python Libraries Version)")
        print("\nThis script converts all HEIC/HEIF files to JPG format using Python libraries.")
        print("⚠️  WARNING: By default, this script DELETES original HEIC files after conversion!")
        print("\nDependencies:")
        print("  - Pillow: pip install Pillow")
        print("  - pillow-heif: pip install pillow-heif")
        print("\nExample usage:")
        print("  python heic_to_jpg_pil.py /path/to/photos")
        print("  python heic_to_jpg_pil.py /path/to/photos --quality 95")
        print("  python heic_to_jpg_pil.py /path/to/photos --keep-original")
        print("  python heic_to_jpg_pil.py /path/to/photos --dry-run")
        print("\nOptions:")
        print("  --quality, -q      JPG quality (1-100, default: 90)")
        print("  --keep-original    Keep original HEIC files (default: delete them)")
        print("  --dry-run         Preview what would be converted")
        print("\nNote: This version uses Python libraries instead of ImageMagick")
        sys.exit(0)
    
    sys.exit(main())
