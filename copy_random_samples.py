#!/usr/bin/env python3
"""
Random File Sampler
Copies one random file from each subdirectory in source folder to corresponding subdirectory in destination folder
"""

import os
import shutil
import random
import argparse
from pathlib import Path

def copy_random_samples(source_dir, dest_dir, extensions=None, dry_run=False):
    """
    Copy one random file from each subdirectory in source to corresponding subdirectory in destination
    
    Args:
        source_dir (str): Source directory containing subdirectories
        dest_dir (str): Destination directory where subdirectories will be created
        extensions (list): List of file extensions to consider (e.g., ['.jpg', '.png', '.bmp'])
        dry_run (bool): If True, only show what would be copied without actually copying
    """
    
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory does not exist: {source_dir}")
        return False
    
    if not source_path.is_dir():
        print(f"❌ Source path is not a directory: {source_dir}")
        return False
    
    # Create destination directory if it doesn't exist
    if not dry_run:
        dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Source: {source_dir}")
    print(f"📂 Destination: {dest_dir}")
    if extensions:
        print(f"🔍 File extensions: {extensions}")
    print(f"🔄 Dry run: {dry_run}")
    print("-" * 50)
    
    copied_count = 0
    skipped_count = 0
    
    # Get all subdirectories in source
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print("❌ No subdirectories found in source directory")
        return False
    
    print(f"📁 Found {len(subdirs)} subdirectories")
    
    for subdir in sorted(subdirs):
        subdir_name = subdir.name
        print(f"\n🔍 Processing subdirectory: {subdir_name}")
        
        # Get all files in subdirectory
        files = []
        for file_path in subdir.iterdir():
            if file_path.is_file():
                # Check extension filter if provided
                if extensions is None or file_path.suffix.lower() in [ext.lower() for ext in extensions]:
                    files.append(file_path)
        
        if not files:
            print(f"  ⚠️  No files found in {subdir_name}")
            skipped_count += 1
            continue
        
        # Select random file
        random_file = random.choice(files)
        print(f"  🎲 Selected random file: {random_file.name} (from {len(files)} files)")
        
        # Create destination subdirectory
        dest_subdir = dest_path / subdir_name
        if not dry_run:
            dest_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        dest_file = dest_subdir / random_file.name
        
        if dry_run:
            print(f"  📋 Would copy: {random_file} → {dest_file}")
        else:
            try:
                shutil.copy2(random_file, dest_file)
                print(f"  ✅ Copied: {random_file.name} → {dest_file}")
                copied_count += 1
            except Exception as e:
                print(f"  ❌ Error copying {random_file.name}: {e}")
                skipped_count += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Summary:")
    print(f"  ✅ Successfully copied: {copied_count} files")
    print(f"  ⚠️  Skipped: {skipped_count} subdirectories")
    print(f"  📁 Total subdirectories processed: {len(subdirs)}")
    
    return copied_count > 0

def main():
    parser = argparse.ArgumentParser(description="Copy one random file from each subdirectory")
    parser.add_argument("source", help="Source directory containing subdirectories")
    parser.add_argument("destination", help="Destination directory")
    parser.add_argument("--extensions", "-e", nargs="+", 
                       help="File extensions to consider (e.g., .jpg .png .bmp)")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Show what would be copied without actually copying")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducible results")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"🎯 Random seed set to: {args.seed}")
    
    success = copy_random_samples(
        args.source, 
        args.destination, 
        args.extensions, 
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
        print("🎯 Random File Sampler")
        print("\nExample usage:")
        print("  python copy_random_samples.py /source/dir /dest/dir")
        print("  python copy_random_samples.py /source/dir /dest/dir --extensions .jpg .png .bmp")
        print("  python copy_random_samples.py /source/dir /dest/dir --dry-run")
        print("  python copy_random_samples.py /source/dir /dest/dir --seed 42")
        print("\nFor your letter dataset:")
        print("  python copy_random_samples.py /home/nvidia10/datasets/lettersdatabase/train /home/nvidia10/datasets/lettersdatabase/sample --extensions .bmp")
        sys.exit(0)
    
    sys.exit(main())
