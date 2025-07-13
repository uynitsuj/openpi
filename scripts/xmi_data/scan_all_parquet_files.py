#!/usr/bin/env python3
"""
Comprehensive scanner to find problematic parquet files.
This will scan ALL files to find any with scalar values instead of arrays.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

def scan_all_parquet_files(dataset_path: Path):
    """Scan all parquet files to find schema inconsistencies."""
    print(f"🔍 Scanning ALL parquet files in: {dataset_path}")
    
    data_dir = dataset_path / "data"
    parquet_files = list(data_dir.rglob("*.parquet"))
    
    print(f"📁 Found {len(parquet_files)} parquet files to scan")
    
    scalar_fields = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    problematic_files = []
    
    for pfile in tqdm(parquet_files, desc="Scanning files"):
        try:
            df = pd.read_parquet(pfile)
            
            file_issues = []
            
            for field in scalar_fields:
                if field in df.columns:
                    sample_val = df[field].iloc[0]
                    
                    # Check if it's in the correct format (array with shape (1,))
                    is_correct_format = (
                        (isinstance(sample_val, list) and len(sample_val) == 1) or
                        (hasattr(sample_val, 'shape') and sample_val.shape == (1,))
                    )
                    
                    if not is_correct_format:
                        file_issues.append({
                            'field': field,
                            'value': sample_val,
                            'type': type(sample_val).__name__,
                            'is_scalar': not hasattr(sample_val, '__len__') or len(sample_val) != 1
                        })
            
            if file_issues:
                problematic_files.append({
                    'file': pfile.name,
                    'path': str(pfile),
                    'issues': file_issues
                })
                
        except Exception as e:
            print(f"❌ Error reading {pfile.name}: {e}")
            problematic_files.append({
                'file': pfile.name,
                'path': str(pfile),
                'error': str(e)
            })
    
    print(f"\n📊 Scan Results:")
    print(f"   Total files scanned: {len(parquet_files)}")
    print(f"   Problematic files: {len(problematic_files)}")
    
    if problematic_files:
        print(f"\n⚠️  Found {len(problematic_files)} problematic files:")
        
        for i, prob_file in enumerate(problematic_files[:10]):  # Show first 10
            print(f"\n  {i+1}. {prob_file['file']}:")
            if 'error' in prob_file:
                print(f"     ❌ Error: {prob_file['error']}")
            else:
                for issue in prob_file['issues']:
                    print(f"     🔴 {issue['field']}: {issue['value']} ({issue['type']})")
        
        if len(problematic_files) > 10:
            print(f"     ... and {len(problematic_files) - 10} more files")
        
        # Show breakdown by issue type
        issue_counts = {}
        for prob_file in problematic_files:
            if 'issues' in prob_file:
                for issue in prob_file['issues']:
                    field = issue['field']
                    if field not in issue_counts:
                        issue_counts[field] = 0
                    issue_counts[field] += 1
        
        print(f"\n📈 Issue breakdown:")
        for field, count in issue_counts.items():
            print(f"   {field}: {count} files")
    
    else:
        print("✅ No problematic files found - all files have correct schema!")
    
    return problematic_files

def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_all_parquet_files.py <dataset_path>")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        sys.exit(1)
    
    problematic_files = scan_all_parquet_files(dataset_path)
    
    if problematic_files:
        print(f"\n💡 To fix these files, you can use:")
        print(f"   python fix_scalar_indices.py --dataset-path {dataset_path}")
    else:
        print(f"\n🎉 Dataset schema is consistent!")

if __name__ == "__main__":
    main() 