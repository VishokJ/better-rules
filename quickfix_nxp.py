#!/usr/bin/env python3
"""
NXP Pin Table and Ordering Info Quickfix Script
Updates existing NXP JSON files with corrected pin table and ordering information
Uses the new fallback logic for PDFs without table of contents
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from functions import extract_pin_table_and_ordering_info_sync

def find_nxp_files() -> List[tuple]:
    """Find matching PDF and JSON file pairs"""
    current_dir = Path.cwd()
    nxp_datasheets_dir = current_dir / "nxp_datasheets"
    nxp_outputs_dir = current_dir / "nxp_outputs"

    if not nxp_datasheets_dir.exists():
        print(f"❌ Source directory not found: {nxp_datasheets_dir}")
        return []

    if not nxp_outputs_dir.exists():
        print(f"❌ Output directory not found: {nxp_outputs_dir}")
        return []

    pairs = []
    for pdf_file in nxp_datasheets_dir.glob("*.pdf"):
        # Extract part ID from filename
        part_id = pdf_file.stem
        # Remove common suffixes
        suffixes_to_remove = [
            '_datasheet', '_ds', '_datasheet_rev', '_rev',
            '_v1', '_v2', '_v3', '_final', '_preliminary',
            '_product_data_sheet', '_data_sheet'
        ]

        for suffix in suffixes_to_remove:
            if part_id.lower().endswith(suffix.lower()):
                part_id = part_id[:-len(suffix)]

        json_file = nxp_outputs_dir / f"{part_id}.json"
        if json_file.exists():
            pairs.append((pdf_file, json_file, part_id))

    return pairs

def update_json_file(pdf_path: Path, json_path: Path, part_id: str) -> bool:
    """Update existing JSON file with new pin table and ordering info"""
    print(f"\n🔄 Processing: {part_id}")
    print(f"📄 PDF: {pdf_path.name}")
    print(f"📝 JSON: {json_path.name}")

    try:
        # Load existing JSON
        with open(json_path, 'r') as f:
            existing_data = json.load(f)

        print(f"📋 Current pin table: {len(existing_data.get('pin', []))} pins")
        print(f"📦 Current ordering info: {'✅' if existing_data.get('ordering_info') else '❌'}")

        # Extract new pin table and ordering info
        print(f"🔍 Extracting pin table and ordering info...")
        start_time = time.time()

        extraction_result = extract_pin_table_and_ordering_info_sync(str(pdf_path))

        extraction_time = time.time() - start_time
        print(f"⏱️  Extraction completed in {extraction_time:.1f}s")

        if "error" in extraction_result:
            print(f"❌ Extraction failed: {extraction_result['error']}")
            return False

        # Update the existing data
        old_pin_count = len(existing_data.get('pin', []))
        new_pin_data = extraction_result.get('pin_table', {})
        new_ordering_data = extraction_result.get('ordering_info', {})

        # Update pin table
        existing_data["pin"] = new_pin_data
        existing_data["ordering_info"] = new_ordering_data

        # Show comparison
        if isinstance(new_pin_data, dict) and new_pin_data:
            if any(isinstance(v, dict) for v in new_pin_data.values()):
                # Multi-package format
                new_pin_count = sum(len(pins) for pins in new_pin_data.values() if isinstance(pins, dict))
                packages = list(new_pin_data.keys())
                print(f"📍 New pin table: {new_pin_count} pins across {len(packages)} packages: {packages}")
            else:
                # Simple format
                new_pin_count = len(new_pin_data)
                print(f"📍 New pin table: {new_pin_count} pins (simple format)")
        else:
            new_pin_count = 0
            print(f"📍 New pin table: No pins found")

        if new_ordering_data:
            position_count = len(new_ordering_data.get('position_options', []))
            print(f"📦 New ordering info: {position_count} position groups")
        else:
            print(f"📦 New ordering info: None found")

        # Save updated JSON
        with open(json_path, 'w') as f:
            json.dump(existing_data, f, indent=2, default=str)

        print(f"💾 Updated: {json_path}")
        print(f"📊 Pin count: {old_pin_count} → {new_pin_count}")

        return True

    except Exception as e:
        print(f"❌ Error updating {part_id}: {str(e)}")
        return False

def main():
    """Main function for NXP quickfix"""

    print("🚀 NXP PIN TABLE & ORDERING INFO QUICKFIX")
    print("=" * 60)
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Find PDF/JSON pairs
    file_pairs = find_nxp_files()
    if not file_pairs:
        print("❌ No matching PDF/JSON pairs found")
        return

    print(f"📄 Found {len(file_pairs)} PDF/JSON pairs to process")

    # Show first few files
    print(f"\n📋 Files to update:")
    for i, (pdf_path, json_path, part_id) in enumerate(file_pairs[:10]):
        print(f"   {i+1:2d}. {part_id} ({pdf_path.name} → {json_path.name})")

    if len(file_pairs) > 10:
        print(f"   ... and {len(file_pairs) - 10} more files")

    print(f"\n⚡ This will:")
    print(f"   • Re-extract pin tables using new fallback logic")
    print(f"   • Re-extract ordering information")
    print(f"   • UPDATE existing JSON files (preserving checklist)")
    print(f"   • Handle PDFs without table of contents (<50 pages)")

    # Process files
    start_time = time.time()
    successful_count = 0
    failed_count = 0

    for i, (pdf_path, json_path, part_id) in enumerate(file_pairs, 1):
        print(f"\n[{i}/{len(file_pairs)}] {part_id}")

        if update_json_file(pdf_path, json_path, part_id):
            successful_count += 1
        else:
            failed_count += 1

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 QUICKFIX SUMMARY")
    print("=" * 60)
    print(f"📄 Total files: {len(file_pairs)}")
    print(f"✅ Successfully updated: {successful_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"📈 Average time per file: {total_time/len(file_pairs):.1f} seconds")

    if failed_count > 0:
        print(f"\n⚠️  {failed_count} files failed to update")
        print(f"💡 Check the console output above for specific error details")

    print(f"\n✅ NXP quickfix completed!")
    print(f"🎯 Updated JSON files now have correct pin tables and ordering info")

if __name__ == "__main__":
    main()