#!/usr/bin/env python3
"""
Script to download processed JSON files from S3 output folder.
Skips files that start with 'an' (case insensitive).
"""

import boto3
import os
import argparse
from botocore.exceptions import ClientError
from pathlib import Path

def download_json_files(s3_path: str, local_folder: str):
    """Download JSON files from S3, skipping files starting with 'an'"""
    
    # Parse S3 path
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    
    bucket_name, prefix = s3_path.split("/", 1)
    if not prefix.endswith("/"):
        prefix += "/"
    
    # Create local output folder
    Path(local_folder).mkdir(exist_ok=True)
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    print(f"🔍 Searching for JSON files in: s3://{bucket_name}/{prefix}")
    print(f"📁 Downloading to: {local_folder}")
    print(f"⚠️  Skipping files starting with 'an' (case insensitive)")
    print()
    
    downloaded_count = 0
    skipped_count = 0
    
    try:
        # List all objects in the S3 prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Only process JSON files
                if not key.lower().endswith('.json'):
                    continue
                
                # Extract filename from key
                filename = os.path.basename(key)
                
                # Skip files starting with 'an' (case insensitive)
                if filename.lower().startswith('an'):
                    print(f"⏭️  Skipping: {filename}")
                    skipped_count += 1
                    continue
                
                # Download the file
                local_path = os.path.join(local_folder, filename)
                
                try:
                    print(f"⬇️  Downloading: {filename}")
                    s3_client.download_file(bucket_name, key, local_path)
                    downloaded_count += 1
                    
                except ClientError as e:
                    print(f"❌ Error downloading {filename}: {e}")
                    continue
    
    except ClientError as e:
        print(f"❌ Error listing S3 objects: {e}")
        return
    
    print()
    print(f"✅ Download complete!")
    print(f"   Downloaded: {downloaded_count} files")
    print(f"   Skipped: {skipped_count} files")
    print(f"   Location: {os.path.abspath(local_folder)}")

def main():
    parser = argparse.ArgumentParser(description="Download processed JSON files from S3")
    parser.add_argument("--s3-path", 
                       default="s3://didy-bucket/St.com/Microcontrollers---microprocessors/output/",
                       help="S3 path to download from")
    parser.add_argument("--output-folder", 
                       default="output",
                       help="Local folder to download to")
    
    args = parser.parse_args()
    
    download_json_files(args.s3_path, args.output_folder)

if __name__ == "__main__":
    main()