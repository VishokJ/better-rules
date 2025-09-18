#!/usr/bin/env python3
"""
Simple test script for pin table page detection
Focus: Just find the continuous page range, not detailed extraction
"""

import sys
import time
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from functions import extract_pin_table_and_ordering_info_sync

def main():
    """Simple test focusing on page range detection"""

    # Check if test.pdf exists
    test_pdf = Path("simple_test.pdf")
    if not test_pdf.exists():
        print("❌ Error: test.pdf not found in current directory")
        return

    print("🚀 SIMPLE PIN TABLE PAGE DETECTION")
    print("=" * 50)
    print(f"📄 Processing: {test_pdf}")
    print(f"⏰ Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    try:
        start_time = time.time()

        # Run the simplified extraction
        result = extract_pin_table_and_ordering_info_sync(str(test_pdf))

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\n⏱️  Processing completed in {processing_time:.2f} seconds")
        print("=" * 50)

        # Display results summary
        if "error" in result:
            print(f"❌ Extraction failed: {result['error']}")
            return

        print("📊 SIMPLE RESULTS:")

        # Show sections found
        if result.get('sections_found'):
            sections = result['sections_found']
            print(f"\n📖 Sections Found:")

            if hasattr(sections, 'pin_descriptions') and sections.pin_descriptions:
                pin_range = sections.pin_descriptions
                print(f"   📍 Pin descriptions: pages {pin_range.start_page}-{pin_range.end_page}")
                print(f"      → Will scan this range for continuous pin table pages")

            if hasattr(sections, 'ordering_information') and sections.ordering_information:
                order_range = sections.ordering_information
                print(f"   📦 Ordering info: pages {order_range.start_page}-{order_range.end_page}")

        # Show pin table detection results
        if result.get('pin_table'):
            print(f"\n📋 Pin Table Results:")
            pin_table = result['pin_table']
            if isinstance(pin_table, dict) and pin_table:
                for package_name, pins in pin_table.items():
                    print(f"   📦 {package_name}: {len(pins)} pins detected")
            else:
                print(f"   ℹ️  Pin table detection completed (VLM processing)")

        # Save results to test.json in processed.json format
        output_file = "simple_test.json"

        # Extract device name from PDF filename (remove .pdf extension)
        device_name = test_pdf.stem

        # Create the same format as processed.json
        output_data = {
            device_name: {
                "filename": test_pdf.name,
                "pin": result.get('pin_table', {}),
                "checklist": []  # Will be populated by rule extraction pipeline
            }
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=str)

        print(f"\n💾 Complete extraction results saved to: {output_file}")
        print(f"📋 Format matches processed.json structure")
        print(f"🔧 Device: {device_name}")
        print(f"📋 Ready for rule extraction pipeline to populate checklist")

        print("\n✅ Simple pin table detection completed!")
        print("🎯 Key insight: The function now focuses on finding the continuous page range")
        print("💡 VLM handles the actual data extraction from the identified pages")

    except Exception as e:
        import traceback
        print(f"\n❌ Detection failed with error: {e}")
        print(f"📋 Full traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()