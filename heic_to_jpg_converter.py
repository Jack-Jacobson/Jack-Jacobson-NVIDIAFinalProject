#!/usr/bin/env python3
"""
HEIC to JPG Converter
Converts all HEIC files to JPG format in every subdirectory of a specified path
"""

import os
import argparse
from pathlib import Path
import subprocess
import sys

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        # Check if ImageMagick is installed
        result = subprocess.run(['magick', '-version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ ImageMagick not found. Please install ImageMagick:")
            print("   sudo apt update && sudo apt install imagemagick")
            return False
        
        print("✅ ImageMagick found")
        return True
        
    except FileNotFoundError:
        try:
            # Try alternative command name
            result = subprocess.run(['convert', '-version'], capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ ImageMagick not found. Please install ImageMagick:")
                print("   sudo apt update && sudo apt install imagemagick")
                return False
            
            print("✅ ImageMagick found (using 'convert' command)")
            return True
            
        except FileNotFoundError:
            print("❌ ImageMagick not found. Please install ImageMagick:")
            print("   sudo apt update && sudo apt install imagemagick")
            print("   or")
            print("   brew install imagemagick  # on macOS")
            return False

def convert_heic_to_jpg(heic_path, jpg_path, quality=90):
    """Convert a single HEIC file to JPG using ImageMagick"""
    try:
        # Try using 'magick' command first
        cmd = ['magick', str(heic_path), '-quality', str(quality), str(jpg_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # Try using 'convert' command as fallback
            cmd = ['convert', str(heic_path), '-quality', str(quality), str(jpg_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return False, result.stderr
        
        return True, None
        
    except Exception as e:
        return False, str(e)

def convert_heic_files(root_dir, quality=90, remove_original=False, dry_run=False):
    """
    Convert all HEIC files to JPG in all subdirectories
    
    Args:
        root_dir (str): Root directory to search for HEIC files
        quality (int): JPG quality (1-100, default 90)
        remove_original (bool): Whether to remove original HEIC files after conversion
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
        success, error = convert_heic_to_jpg(heic_file, jpg_file, quality)
        
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
    parser = argparse.ArgumentParser(description="Convert all HEIC files to JPG in directory tree")
    parser.add_argument("directory", help="Root directory to search for HEIC files")
    parser.add_argument("--quality", "-q", type=int, default=90, choices=range(1, 101),
                       help="JPG quality (1-100, default: 90)")
    parser.add_argument("--remove-original", "-r", action="store_true",
                       help="Remove original HEIC files after successful conversion")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Show what would be converted without actually converting")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    success = convert_heic_files(
        args.directory,
        args.quality,
        args.remove_original,
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
        print("📱 HEIC to JPG Converter")
        print("\nThis script converts all HEIC/HEIF files to JPG format in a directory tree.")
        print("\nDependencies:")
        print("  - ImageMagick (sudo apt install imagemagick)")
        print("\nExample usage:")
        print("  python heic_to_jpg_converter.py /path/to/photos")
        print("  python heic_to_jpg_converter.py /path/to/photos --quality 95")
        print("  python heic_to_jpg_converter.py /path/to/photos --remove-original")
        print("  python heic_to_jpg_converter.py /path/to/photos --dry-run")
        print("\nOptions:")
        print("  --quality, -q      JPG quality (1-100, default: 90)")
        print("  --remove-original  Remove original HEIC files after conversion")
        print("  --dry-run         Preview what would be converted")
        sys.exit(0)
    
    sys.exit(main())
