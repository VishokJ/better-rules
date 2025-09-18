#!/usr/bin/env python3
"""
Local PDF processing script to extract pin tables, ordering info, and rules from datasheets folder.
Results are saved to outputs folder.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import sys
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from functions import extract_pin_table_and_ordering_info_sync
from extraction import StandalonePDFRuleExtractor

def ensure_directories():
    """Create datasheets and outputs directories if they don't exist"""
    datasheets_dir = Path("datasheets")
    outputs_dir = Path("outputs")

    datasheets_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    return datasheets_dir, outputs_dir

def get_pdf_files(datasheets_dir: Path) -> List[Path]:
    """Get all PDF files from datasheets directory"""
    pdf_files = list(datasheets_dir.glob("*.pdf")) + list(datasheets_dir.glob("*.PDF"))
    return sorted(pdf_files)

async def process_single_pdf(pdf_path: Path, outputs_dir: Path) -> Dict[str, Any]:
    """Process a single PDF and extract all required information"""
    print(f"Processing: {pdf_path.name}")

    # Step 1: Extract pin table and ordering info
    try:
        pin_ordering_result = extract_pin_table_and_ordering_info_sync(str(pdf_path))
        pin_table = pin_ordering_result.get('pin_table', [])
        ordering_info = pin_ordering_result.get('ordering_info', {})

        print(f"  ✅ Pin table extracted: {len(pin_table)} pins")
        print(f"  ✅ Ordering info extracted")

    except Exception as e:
        print(f"  ❌ Error extracting pin table/ordering: {e}")
        pin_table = []
        ordering_info = {}

    # Step 2: Extract design rules
    try:
        # Initialize rule extractor with correct organization name for SCHEMATIC_chunks table
        rule_extractor = StandalonePDFRuleExtractor(
            organization_name="SCHEMATIC"
        )

        # Convert pin table to string format for rule extraction
        pin_table_str = json.dumps(pin_table, indent=2) if pin_table else None

        rules = await rule_extractor.extract_rules_from_pdf(
            pdf_name=pdf_path.name,
            pin_table_data=pin_table_str
        )

        print(f"  ✅ Design rules extracted: {len(rules)} rules")

    except Exception as e:
        print(f"  ❌ Error extracting rules: {e}")
        rules = []

    # Prepare final result
    result = {
        "source_file": str(pdf_path),
        "device_name": pdf_path.stem,
        "pin_table": pin_table,
        "ordering_info": ordering_info,
        "extracted_rules": rules,
        "processing_summary": {
            "pin_count": len(pin_table),
            "rule_count": len(rules),
            "has_ordering_info": bool(ordering_info)
        }
    }

    # Save individual result
    output_file = outputs_dir / f"{pdf_path.stem}_processed.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  💾 Saved: {output_file.name}")

    return result

async def main():
    """Main processing function"""
    print("🚀 Better Rules - Local PDF Processor")
    print("=" * 50)

    # Setup directories
    datasheets_dir, outputs_dir = ensure_directories()

    # Find PDF files
    pdf_files = get_pdf_files(datasheets_dir)

    if not pdf_files:
        print(f"❌ No PDF files found in {datasheets_dir}/")
        print(f"   Please place PDF files in the datasheets/ folder")
        return

    print(f"📁 Found {len(pdf_files)} PDF files in {datasheets_dir}/")
    print(f"📁 Results will be saved to {outputs_dir}/")
    print()

    # Process each PDF
    all_results = []
    failed_files = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            result = await process_single_pdf(pdf_path, outputs_dir)
            all_results.append(result)

        except Exception as e:
            print(f"  ❌ Failed to process {pdf_path.name}: {e}")
            failed_files.append(str(pdf_path))

    # Save summary
    summary = {
        "total_files": len(pdf_files),
        "successful": len(all_results),
        "failed": len(failed_files),
        "failed_files": failed_files,
        "results": all_results
    }

    summary_file = outputs_dir / "processing_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print final summary
    print()
    print("📊 PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Total files: {len(pdf_files)}")
    print(f"Successful: {len(all_results)}")
    print(f"Failed: {len(failed_files)}")
    print(f"📁 All results saved in: {outputs_dir}/")
    print(f"📄 Summary saved as: {summary_file}")

    if failed_files:
        print("\n❌ Failed files:")
        for failed_file in failed_files:
            print(f"  - {failed_file}")

if __name__ == "__main__":
    # Check environment
    missing_vars = []
    if not os.getenv('GOOGLE_API_KEY') and not os.getenv('OPENAI_API_KEY'):
        missing_vars.append("GOOGLE_API_KEY or OPENAI_API_KEY")
    if not os.getenv('MONGODB_URI'):
        missing_vars.append("MONGODB_URI")

    if missing_vars:
        print("❌ Error: Missing required environment variables!")
        print("   Please set the following in your .env file:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nExample .env file:")
        print("GOOGLE_API_KEY=your_google_api_key")
        print("MONGODB_URI=mongodb://localhost:27017/your_database")
        sys.exit(1)

    asyncio.run(main())