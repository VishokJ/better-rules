#!/usr/bin/env python3
"""
Script to continue processing documents that haven't been completed yet.
Reads logs to identify processed documents and continues with remaining ones.
"""

import asyncio
import argparse
import re
import os
import sys
from typing import Set, List
from main import AsyncPipelineManager

def parse_logs(log_file_path: str) -> Set[str]:
    """Parse log file to find successfully completed documents"""
    completed_docs = set()
    
    if not os.path.exists(log_file_path):
        print(f"⚠️  Log file not found: {log_file_path}")
        return completed_docs
    
    print(f"📖 Parsing logs from: {log_file_path}")
    
    try:
        with open(log_file_path, 'r') as f:
            for line in f:
                # Look for successful completion messages
                if "✅ Completed" in line:
                    # Extract document path from log line
                    # Format: "✅ Completed (X/Y): path/to/document.pdf"
                    match = re.search(r'✅ Completed \(\d+/\d+\): (.+\.pdf)', line)
                    if match:
                        doc_path = match.group(1)
                        completed_docs.add(doc_path)
                        
                # Also look for successful rule generation completion
                elif "Rule generation completed successfully" in line:
                    match = re.search(r'Rule generation completed successfully for (.+\.pdf)', line)
                    if match:
                        doc_path = match.group(1)
                        completed_docs.add(doc_path)
    
    except Exception as e:
        print(f"❌ Error reading log file: {e}")
        return completed_docs
    
    print(f"📋 Found {len(completed_docs)} completed documents in logs")
    return completed_docs

def get_all_documents(s3_path: str, filter_prefix: str = "") -> List[str]:
    """Get all documents that should be processed"""
    import boto3
    from botocore.exceptions import ClientError
    
    print(f"🔍 Getting all documents from: {s3_path}")
    
    # Extract bucket and path from s3:// URL
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]  # Remove s3:// prefix
    
    bucket_name, base_prefix = s3_path.split("/", 1)
    
    # Use boto3 directly to list all PDFs
    s3_client = boto3.client('s3')
    all_docs = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.lower().endswith('.pdf'):
                        # Remove base prefix to get relative path
                        relative_path = key
                        all_docs.append(relative_path)
    
    except ClientError as e:
        print(f"❌ Error listing S3 objects: {e}")
        return []
    
    print(f"📂 Found {len(all_docs)} total documents")
    return all_docs

async def continue_processing(remaining_docs: List[str], workers: int, s3_path: str):
    """Continue processing with remaining documents"""
    if not remaining_docs:
        print("🎉 All documents have been processed!")
        return
    
    print(f"🚀 Starting processing of {len(remaining_docs)} remaining documents with {workers} workers")
    print(f"📄 Sample remaining documents:")
    for i, doc in enumerate(remaining_docs[:5]):
        print(f"   {i+1}. {doc}")
    if len(remaining_docs) > 5:
        print(f"   ... and {len(remaining_docs) - 5} more")
    
    # Extract path info from s3_path  
    if s3_path.startswith("s3://"):
        s3_path = s3_path[5:]
    bucket_name, base_prefix = s3_path.split("/", 1)
    
    # Initialize pipeline (organization_name should be "ST" for STMicroelectronics)
    pipeline = AsyncPipelineManager(
        organization_name="ST",  # Fixed to use correct organization name
        base_prefix=base_prefix,
        max_workers=workers
    )
    
    try:
        # Process remaining documents
        await pipeline.process_streaming_pipeline(remaining_docs)
        
        print(f"✅ Finished processing {len(remaining_docs)} documents")
        pipeline.print_summary()
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
    finally:
        pipeline.cleanup()

def main():
    parser = argparse.ArgumentParser(description="Continue processing unfinished documents")
    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
    parser.add_argument("--s3-path", required=True, help="S3 path to process")
    parser.add_argument("--filter-prefix", default="", help="Filter prefix for documents")
    parser.add_argument("--log-file", default="/var/log/better-rules/app.log", help="Path to log file")
    
    args = parser.parse_args()
    
    print("🔄 Continuing Better Rules processing...")
    print(f"   Workers: {args.workers}")
    print(f"   S3 Path: {args.s3_path}")
    print(f"   Filter: {args.filter_prefix or 'None'}")
    print(f"   Log file: {args.log_file}")
    print()
    
    # Parse logs to find completed documents
    completed_docs = parse_logs(args.log_file)
    
    # Get all documents that should be processed
    all_docs = get_all_documents(args.s3_path, args.filter_prefix)
    
    # Find remaining documents
    remaining_docs = [doc for doc in all_docs if doc not in completed_docs]
    
    print(f"📊 Processing Summary:")
    print(f"   Total documents: {len(all_docs)}")
    print(f"   Completed: {len(completed_docs)}")
    print(f"   Remaining: {len(remaining_docs)}")
    print()
    
    if remaining_docs:
        # Continue processing
        asyncio.run(continue_processing(remaining_docs, args.workers, args.s3_path))
    else:
        print("🎉 All documents have been processed successfully!")

if __name__ == "__main__":
    main()