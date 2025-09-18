#!/usr/bin/env python3
"""
Test script to run the extraction pipeline on test.pdf
Compares Camelot vs pdfplumber table extraction performance
"""

import asyncio
import sys
import os
from pathlib import Path
import time

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from functions import extract_pin_table_and_ordering_info_sync
import json

def main():
    """Main function to test extraction pipeline"""

    # Check if test.pdf exists
    test_pdf = Path("test.pdf")
    if not test_pdf.exists():
        print("❌ Error: test.pdf not found in current directory")
        print("Please place your test PDF file as 'test.pdf' in this directory")
        return

    print("🚀 EXTRACTION PIPELINE TEST")
    print("=" * 50)
    print(f"📄 Processing: {test_pdf}")
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    try:
        print("\n🔍 Starting extraction process...")
        start_time = time.time()

        # Run the extraction (synchronous version to avoid async complexity)
        result = extract_pin_table_and_ordering_info_sync(str(test_pdf))

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n⏱️  Processing completed in {processing_time:.2f} seconds")
        print("=" * 50)

        # Display results summary
        if "error" in result:
            print(f"❌ Extraction failed: {result['error']}")
            return

        print("📊 EXTRACTION RESULTS SUMMARY:")
        print(f"   📋 Pin table packages found: {len(result.get('pin_table', {}))}")
        print(f"   📦 Ordering info available: {'✅' if result.get('ordering_info') else '❌'}")

        # Show pin table packages
        if result.get('pin_table'):
            print(f"\n📋 Pin Table Packages:")
            for package_name, pins in result['pin_table'].items():
                print(f"   • {package_name}: {len(pins)} pins")

        # Show ordering info
        if result.get('ordering_info'):
            ordering = result['ordering_info']
            print(f"\n📦 Ordering Information:")
            print(f"   • Manufacturer: {ordering.get('manufacturer', 'N/A')}")
            print(f"   • Component type: {ordering.get('component_type', 'N/A')}")
            print(f"   • Position options: {len(ordering.get('position_options', []))} positions")
            print(f"   • Pages processed: {ordering.get('pages_processed', 'N/A')}")

        # Show sections found
        if result.get('sections_found'):
            sections = result['sections_found']
            print(f"\n📖 Sections Found:")

            if hasattr(sections, 'pin_descriptions') and sections.pin_descriptions:
                print(f"   • Pin descriptions: pages {sections.pin_descriptions.start_page}-{sections.pin_descriptions.end_page}")

            if hasattr(sections, 'ordering_information') and sections.ordering_information:
                print(f"   • Ordering info: pages {sections.ordering_information.start_page}-{sections.ordering_information.end_page}")

        # Save results to file
        output_file = "test_extraction_results.json"
        with open(output_file, "w") as f:
            # Convert result to JSON-serializable format
            json_result = {}
            for key, value in result.items():
                if hasattr(value, '__dict__'):
                    # Convert Pydantic models to dict
                    json_result[key] = value.__dict__ if hasattr(value, '__dict__') else str(value)
                else:
                    json_result[key] = value

            json.dump(json_result, f, indent=2, default=str)

        print(f"\n💾 Full results saved to: {output_file}")

        # Check for comparison file
        comparison_file = Path("table_extraction_comparison.json")
        if comparison_file.exists():
            print(f"📊 Table comparison data saved to: {comparison_file}")

            # Load and show comparison summary
            with open(comparison_file) as f:
                comparison_data = json.load(f)

            print(f"\n🏆 CAMELOT vs PDFPLUMBER SUMMARY:")
            camelot_wins = 0
            pdfplumber_wins = 0
            ties = 0

            for page_num, data in comparison_data.items():
                camelot_score = data['camelot_score']
                pdfplumber_score = data['pdfplumber_score']

                if camelot_score > pdfplumber_score:
                    camelot_wins += 1
                elif pdfplumber_score > camelot_score:
                    pdfplumber_wins += 1
                else:
                    ties += 1

            total_pages = len(comparison_data)
            print(f"   📄 Pages analyzed: {total_pages}")
            print(f"   🏆 Camelot wins: {camelot_wins}")
            print(f"   🏆 pdfplumber wins: {pdfplumber_wins}")
            print(f"   🤝 Ties: {ties}")

            if camelot_wins > pdfplumber_wins:
                overall_winner = "🏆 Overall winner: Camelot"
            elif pdfplumber_wins > camelot_wins:
                overall_winner = "🏆 Overall winner: pdfplumber"
            else:
                overall_winner = "🤝 Overall result: Tie"

            print(f"   {overall_winner}")

        print("\n✅ Extraction test completed successfully!")

    except Exception as e:
        import traceback
        print(f"\n❌ Extraction failed with error: {e}")
        print(f"📋 Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()