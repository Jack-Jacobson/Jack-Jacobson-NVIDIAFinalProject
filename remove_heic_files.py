#!/usr/bin/env python3
"""
HEIC File Remover
Removes all HEIC/HEIF files in a directory and its subdirectories
"""

import os
import argparse
from pathlib import Path
import sys

def remove_heic_files(root_dir, dry_run=False):
    """
    Remove all HEIC/HEIF files in directory tree
    
    Args:
        root_dir (str): Root directory to search for HEIC files
        dry_run (bool): If True, only show what would be removed without actually removing
    """
    
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ Directory does not exist: {root_dir}")
        return False
    
    if not root_path.is_dir():
        print(f"❌ Path is not a directory: {root_dir}")
        return False
    
    print(f"📂 Searching for HEIC files in: {root_dir}")
    print(f"🔄 Dry run: {dry_run}")
    if not dry_run:
        print("⚠️  WARNING: This will PERMANENTLY DELETE all HEIC files!")
    print("-" * 60)
    
    # Find all HEIC files recursively
    heic_extensions = ['.heic', '.HEIC', '.heif', '.HEIF']
    heic_files = []
    
    for ext in heic_extensions:
        heic_files.extend(root_path.rglob(f"*{ext}"))
    
    if not heic_files:
        print("❌ No HEIC/HEIF files found in the directory tree")
        return False
    
    print(f"📱 Found {len(heic_files)} HEIC/HEIF files to remove")
    print()
    
    removed_count = 0
    failed_count = 0
    
    for heic_file in heic_files:
        # Get relative path for display
        try:
            rel_heic = heic_file.relative_to(root_path)
        except ValueError:
            rel_heic = heic_file
        
        print(f"🗑️  Processing: {rel_heic}")
        
        if dry_run:
            print(f"  📋 Would remove: {rel_heic}")
            removed_count += 1
            continue
        
        # Remove the file
        try:
            heic_file.unlink()
            print(f"  ✅ Removed: {rel_heic}")
            removed_count += 1
        except Exception as e:
            print(f"  ❌ Failed to remove: {e}")
            failed_count += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Removal Summary:")
    print(f"  ✅ Successfully removed: {removed_count} files")
    print(f"  ❌ Failed removals: {failed_count} files")
    print(f"  📱 Total HEIC files found: {len(heic_files)}")
    
    return removed_count > 0 or dry_run

def main():
    parser = argparse.ArgumentParser(description="Remove all HEIC/HEIF files in directory tree")
    parser.add_argument("directory", help="Root directory to search for HEIC files")
    parser.add_argument("--dry-run", "-d", action="store_true",
                       help="Show what would be removed without actually removing")
    parser.add_argument("--confirm", "-c", action="store_true",
                       help="Skip confirmation prompt and proceed with deletion")
    
    args = parser.parse_args()
    
    # Safety confirmation unless dry run or explicit confirm
    if not args.dry_run and not args.confirm:
        print("⚠️  WARNING: This will PERMANENTLY DELETE all HEIC/HEIF files in:")
        print(f"   {args.directory}")
        print("\nThis action cannot be undone!")
        response = input("\nAre you sure you want to proceed? (type 'yes' to confirm): ")
        
        if response.lower() != 'yes':
            print("❌ Operation cancelled by user")
            return 0
    
    success = remove_heic_files(args.directory, args.dry_run)
    
    if success:
        if args.dry_run:
            print("\n🎉 Dry run completed successfully!")
        else:
            print("\n🎉 Files removed successfully!")
    else:
        print("\n💥 Operation failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    # Example usage when run directly
    import sys
    
    if len(sys.argv) == 1:
        print("🗑️  HEIC File Remover")
        print("\nThis script removes all HEIC/HEIF files in a directory tree.")
        print("⚠️  WARNING: This will PERMANENTLY DELETE files!")
        print("\nExample usage:")
        print("  python remove_heic_files.py /path/to/photos --dry-run")
        print("  python remove_heic_files.py /path/to/photos --confirm")
        print("  python remove_heic_files.py /path/to/photos")
        print("\nOptions:")
        print("  --dry-run, -d    Preview what would be removed")
        print("  --confirm, -c    Skip confirmation prompt")
        print("\nSafety features:")
        print("  - Requires explicit confirmation by default")
        print("  - Dry run mode to preview actions")
        print("  - Detailed logging of all operations")
        sys.exit(0)
    
    sys.exit(main())
