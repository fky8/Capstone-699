"""
Dataset Restore Utility
-----------------------
Backup and restore original datasets before Trump data processing.

This utility allows you to:
1. Backup the current state of data files
2. Restore datasets to their original state (pre-Trump processing)
3. Clean Trump-related files to start fresh

Usage:
    # Backup current datasets
    python restore_datasets.py --backup
    
    # Restore to original state (removes Trump files)
    python restore_datasets.py --restore
    
    # Clean only Trump files (keep backups)
    python restore_datasets.py --clean-trump
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List
import os


class DatasetRestore:
    """Manage dataset backups and restoration."""
    
    def __init__(self, data_dir: str = "data_files"):
        self.data_dir = Path(data_dir)
        self.backup_dir = self.data_dir / "backups"
        
        # Files to backup (original datasets, pre-Trump)
        self.core_files = [
            "master_raw_mags_1m.csv",
            "features_garch.csv",
            "features_news.csv",
            "features_combined.csv",
            "labels.csv",
            "news_raw.csv",
            "news_sentiment.csv"
        ]
        
        # Trump-related files to remove during restore
        self.trump_files = [
            "trump_raw_*.csv",
            "trump_sentiment.csv",
            "features_trump.csv"
        ]
    
    def backup(self, tag: str = None) -> None:
        """
        Create backup of current datasets.
        
        Args:
            tag: Optional tag for backup (default: timestamp)
        """
        if tag is None:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        backup_path = self.backup_dir / f"backup_{tag}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📦 Creating backup: {backup_path}")
        print("=" * 80)
        
        backed_up = []
        missing = []
        
        for filename in self.core_files:
            source = self.data_dir / filename
            
            if source.exists():
                dest = backup_path / filename
                shutil.copy2(source, dest)
                size_mb = source.stat().st_size / (1024 * 1024)
                backed_up.append(f"{filename} ({size_mb:.2f} MB)")
                print(f"  ✓ {filename} ({size_mb:.2f} MB)")
            else:
                missing.append(filename)
                print(f"  ⚠️  {filename} (not found)")
        
        # Create manifest
        manifest_path = backup_path / "MANIFEST.txt"
        with open(manifest_path, 'w') as f:
            f.write(f"Backup created: {datetime.now()}\n")
            f.write(f"Tag: {tag}\n\n")
            f.write("Backed up files:\n")
            for item in backed_up:
                f.write(f"  - {item}\n")
            if missing:
                f.write("\nMissing files:\n")
                for item in missing:
                    f.write(f"  - {item}\n")
        
        print(f"\n💾 Backup complete: {len(backed_up)} files")
        print(f"📄 Manifest: {manifest_path}")
    
    def list_backups(self) -> List[Path]:
        """List available backups."""
        if not self.backup_dir.exists():
            return []
        
        backups = sorted(self.backup_dir.glob("backup_*"))
        return backups
    
    def restore(self, backup_tag: str = "latest") -> None:
        """
        Restore datasets from backup.
        
        Args:
            backup_tag: Tag of backup to restore (default: latest)
        """
        backups = self.list_backups()
        
        if not backups:
            print("❌ No backups found")
            print(f"💡 Create a backup first: python restore_datasets.py --backup")
            return
        
        # Select backup
        if backup_tag == "latest":
            backup_path = backups[-1]
        else:
            backup_path = self.backup_dir / f"backup_{backup_tag}"
            if not backup_path.exists():
                print(f"❌ Backup not found: {backup_tag}")
                print(f"\nAvailable backups:")
                for bp in backups:
                    print(f"  - {bp.name}")
                return
        
        print(f"🔄 Restoring from: {backup_path}")
        print("=" * 80)
        
        # Check manifest
        manifest_path = backup_path / "MANIFEST.txt"
        if manifest_path.exists():
            print("\n📄 Backup manifest:")
            with open(manifest_path, 'r') as f:
                print(f.read())
        
        # Confirm
        print("\n⚠️  This will:")
        print("  1. Restore original datasets from backup")
        print("  2. Remove all Trump-related files")
        print("  3. Remove features_combined.csv (needs regeneration)")
        
        response = input("\nContinue? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Restore cancelled")
            return
        
        # Restore files
        print("\n📥 Restoring files...")
        restored = []
        
        for filename in self.core_files:
            source = backup_path / filename
            dest = self.data_dir / filename
            
            if source.exists():
                shutil.copy2(source, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                restored.append(filename)
                print(f"  ✓ {filename} ({size_mb:.2f} MB)")
        
        # Clean Trump files
        print("\n🧹 Cleaning Trump files...")
        self._clean_trump_files()
        
        print(f"\n✅ Restore complete: {len(restored)} files restored")
        print(f"💡 To regenerate features: make or run pipeline scripts")
    
    def _clean_trump_files(self) -> None:
        """Remove Trump-related files."""
        removed = []
        
        # Remove Trump data files
        for pattern in self.trump_files:
            for file_path in self.data_dir.glob(pattern):
                file_path.unlink()
                removed.append(file_path.name)
                print(f"  🗑️  {file_path.name}")
        
        if not removed:
            print("  (no Trump files found)")
    
    def clean_trump(self) -> None:
        """Clean Trump files without restoring backups."""
        print("🧹 Cleaning Trump files...")
        print("=" * 80)
        
        response = input("Remove all Trump-related data files? [y/N]: ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
        
        self._clean_trump_files()
        print("\n✅ Trump files removed")
        print("💡 To regenerate: python trump_data.py && python trump_sentiment.py && python trump_features.py")
    
    def show_status(self) -> None:
        """Show current status of datasets and backups."""
        print("📊 Dataset Status")
        print("=" * 80)
        
        # Core files
        print("\n📁 Core datasets:")
        for filename in self.core_files:
            file_path = self.data_dir / filename
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                print(f"  ✓ {filename:30s} {size_mb:8.2f} MB  {mtime.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"  ✗ {filename:30s} (missing)")
        
        # Trump files
        print("\n📱 Trump datasets:")
        trump_found = False
        for pattern in self.trump_files:
            for file_path in self.data_dir.glob(pattern):
                trump_found = True
                size_mb = file_path.stat().st_size / (1024 * 1024)
                mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                print(f"  ✓ {file_path.name:30s} {size_mb:8.2f} MB  {mtime.strftime('%Y-%m-%d %H:%M')}")
        
        if not trump_found:
            print("  (no Trump files)")
        
        # Backups
        print("\n💾 Backups:")
        backups = self.list_backups()
        if backups:
            for backup_path in backups:
                manifest_path = backup_path / "MANIFEST.txt"
                if manifest_path.exists():
                    with open(manifest_path, 'r') as f:
                        first_line = f.readline().strip()
                    print(f"  📦 {backup_path.name:30s} - {first_line}")
                else:
                    print(f"  📦 {backup_path.name}")
        else:
            print("  (no backups)")


def main():
    parser = argparse.ArgumentParser(
        description="Backup and restore datasets for Trump processing"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--backup',
        action='store_true',
        help="Create backup of current datasets"
    )
    group.add_argument(
        '--restore',
        metavar='TAG',
        nargs='?',
        const='latest',
        help="Restore datasets from backup (default: latest)"
    )
    group.add_argument(
        '--clean-trump',
        action='store_true',
        help="Remove Trump files only (keep backups)"
    )
    group.add_argument(
        '--status',
        action='store_true',
        help="Show current status of datasets and backups"
    )
    
    parser.add_argument(
        '--tag',
        type=str,
        help="Tag for backup (used with --backup)"
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data_files',
        help="Data directory (default: data_files)"
    )
    
    args = parser.parse_args()
    
    restore = DatasetRestore(data_dir=args.data_dir)
    
    if args.backup:
        restore.backup(tag=args.tag)
    elif args.restore:
        restore.restore(backup_tag=args.restore)
    elif args.clean_trump:
        restore.clean_trump()
    elif args.status:
        restore.show_status()


if __name__ == "__main__":
    main()
