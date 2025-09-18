import fitz
import re
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from openai import AsyncOpenAI
import time
from PIL import Image
import io
from pydantic import BaseModel
import pdfplumber
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

load_dotenv()

# Global semaphore for PDF operations to prevent thread safety issues while allowing some concurrency
# Increase to 16 concurrent PDF operations for better throughput
PDF_SEMAPHORE = threading.Semaphore(16)

RELEVANT_SECTIONS = [
    # for pin table, package info
    "Pin Descriptions",
    "Ordering Information",

    # recommended operating conditions
    "Recommended Operating Conditions",

    # for power supply requirements
    "Power Supply",
    "Electrical Characteristics",
    "Power Consumption",
    "Voltage Specifications",
    "Current Requirements",

    # for interface requirements
    "Interface Requirements",
    "Interface Specifications",
    "Digital I/O",
    "Analog Specifications",

    # component & circuit sections
    "External Components",
    "Crystal Oscillator",
    "Decoupling Capacitors",
    "Reference Design",
    "Application Circuits",

    # operational sections
    "Operating Conditions",
    "ESD Protection",
    "EMC Considerations"

]

class PageRange(BaseModel):
    start_page: int
    end_page: int

class RelevantSections(BaseModel):
    pin_descriptions: Optional[PageRange] = None
    ordering_information: Optional[PageRange] = None

    recommended_operating_conditions: Optional[PageRange] = None

    power_supply: Optional[PageRange] = None
    electrical_characteristics: Optional[PageRange] = None
    power_consumption: Optional[PageRange] = None
    voltage_specifications: Optional[PageRange] = None
    current_requirements: Optional[PageRange] = None

    interface_requirements: Optional[PageRange] = None
    interface_specifications: Optional[PageRange] = None
    digital_io: Optional[PageRange] = None
    analog_specifications: Optional[PageRange] = None

    external_components: Optional[PageRange] = None
    crystal_oscillator: Optional[PageRange] = None
    decoupling_capacitors: Optional[PageRange] = None
    reference_design: Optional[PageRange] = None
    application_circuits: Optional[PageRange] = None

    operating_conditions: Optional[PageRange] = None
    esd_protection: Optional[PageRange] = None
    emc_considerations: Optional[PageRange] = None

class ComponentOrderingScheme(BaseModel):
    manufacturer: Optional[str] = None
    component_type: Optional[str] = None
    position_options: List[List[Dict[str, str]]]
    position_descriptions: List[str] = []

class PinTableResult(BaseModel):
    packages: Dict[str, Dict[str, Dict[str, str]]]  # package_name -> {pin_number -> {column_name -> value}}

class SimplePinTableResult(BaseModel):
    pins: Dict[str, Dict[str, str]]  # pin_number -> {column_name -> value}

def extract_toc(pdf_path: Path):
    with PDF_SEMAPHORE:  # Allow limited concurrent PyMuPDF operations
        doc = fitz.open(pdf_path)
        try:
            toc = doc.get_toc()
            return toc
        finally:
            doc.close()  # Explicitly close document

def find_relevant_sections_sync(toc: List[Tuple[int, str, int]]) -> RelevantSections:
    """Synchronous version of find_relevant_sections for thread pool execution"""
    time_start = time.time()
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    toc_str = "\n".join([f"Level {level}: {title} (Page {page})" for level, title, page in toc])
    
    prompt = f"""
    Analyze this Table of Contents from a semiconductor datasheet and find sections that match or contain these relevant topics:
    - Pin Descriptions (look for sections about pin assignments, pin descriptions, pin information, etc.)
    - Ordering Information (look for sections about ordering, part numbers, package options, etc.)

    Table of Contents:
    {toc_str}

    For each relevant topic found, determine the start page and end page (where the next section begins).
    If a topic is not found, set it to null.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=RelevantSections,
        reasoning_effort="minimal",
    )
    
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    
    result = response.choices[0].message.parsed
    return result

async def find_relevant_sections(toc: List[Tuple[int, str, int]]) -> Dict[str, Optional[List[int]]]:
    time_start = time.time()
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    toc_str = "\n".join([f"Level {level}: {title} (Page {page})" for level, title, page in toc])
    
    prompt = f"""
    Analyze this Table of Contents from a semiconductor datasheet and find sections that match or contain these relevant topics:
    - Pin Descriptions (look for sections about pin assignments, pin descriptions, pin information, etc.)
    - Ordering Information (look for sections about ordering, part numbers, package options, etc.)

    Table of Contents:
    {toc_str}

    For each relevant topic found, determine the start page and end page (where the next section begins).
    If a topic is not found, set it to null.
    """

    response = await client.beta.chat.completions.parse(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format=RelevantSections,
        reasoning_effort="minimal",
    )
    
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")
    
    result = response.choices[0].message.parsed

    return result

def take_screenshots(pdf_path: Path, start_page: int, end_page: int):
    with PDF_SEMAPHORE:  # Allow limited concurrent PyMuPDF operations
        doc = fitz.open(pdf_path)
        try:
            screenshots = []

            for page_num in range(start_page - 1, end_page):
                if page_num >= len(doc):
                    break
                
                page = doc[page_num]
                # Get page as image with high resolution
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                screenshots.append(img)
                
                print(f"Captured screenshot of page {page_num + 1}")
            
            return screenshots
        finally:
            doc.close()  # Explicitly close document

def process_ordering_information_sync(pdf_path: Path, start_page: int, end_page: int):
    """Synchronous version of process_ordering_information for thread pool execution"""
    screenshots = take_screenshots(pdf_path, start_page, end_page)
    print(f"Captured {len(screenshots)} ordering information pages")
    
    # Extract actual ordering scheme using VLM (synchronous)
    if screenshots:
        try:
            print(f"Extracting ordering scheme from {len(screenshots)} images...")
            ordering_scheme = extract_ordering_scheme_vlm_sync(screenshots)
            
            # Convert to dict for JSON serialization
            if ordering_scheme:
                return {
                    "manufacturer": ordering_scheme.manufacturer,
                    "component_type": ordering_scheme.component_type,
                    "position_options": ordering_scheme.position_options,
                    "position_descriptions": ordering_scheme.position_descriptions,
                    "pages_processed": len(screenshots),
                    "start_page": start_page,
                    "end_page": end_page
                }
            else:
                print("Warning: No ordering scheme extracted")
                return {}
                
        except Exception as e:
            print(f"Error extracting ordering scheme: {e}")
            return {}
    
    return {}

async def process_ordering_information(pdf_path: Path, start_page: int, end_page: int):
    screenshots = take_screenshots(pdf_path, start_page, end_page)
    print(f"Captured {len(screenshots)} ordering information pages")
    
    # Extract actual ordering scheme using VLM
    if screenshots:
        try:
            print(f"Extracting ordering scheme from {len(screenshots)} images...")
            ordering_scheme = await extract_ordering_scheme_vlm(screenshots)
            
            # Convert to dict for JSON serialization
            if ordering_scheme:
                return {
                    "manufacturer": ordering_scheme.manufacturer,
                    "component_type": ordering_scheme.component_type,
                    "position_options": ordering_scheme.position_options,
                    "position_descriptions": ordering_scheme.position_descriptions,
                    "pages_processed": len(screenshots),
                    "start_page": start_page,
                    "end_page": end_page
                }
            else:
                print("Warning: No ordering scheme extracted")
                return {}
                
        except Exception as e:
            print(f"Error extracting ordering scheme: {e}")
            return {}
    
    return {}

def extract_ordering_scheme_vlm_sync(images: List[Image.Image]) -> ComponentOrderingScheme:
    """Synchronous version of extract_ordering_scheme_vlm for thread pool execution"""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    
    prompt = """
    You are analyzing component datasheets to extract ordering/part numbering schemes. These schemes define how manufacturers structure their part numbers with different options at each position.

    TASK OVERVIEW:
    Extract the complete part numbering structure from any ordering information, part number breakdown, or ordering code table you find in the images. The information may be spread across multiple images.

    ANALYSIS STEPS:
    1. IDENTIFY the ordering scheme section (often titled "Ordering Information", "Part Numbering", "Product Selection", etc.)
    2. LOCATE the hierarchical breakdown showing each position's meaning
    3. EXTRACT all possible options for each position in the part number
    4. CAPTURE position descriptions (e.g., "Package Type", "Memory Size", etc.)

    EXTRACTION RULES:
    - Include EVERY possible option code for each position
    - Maintain exact position order as shown in the source
    - Use exact text/codes from the document (preserve case, numbers, special characters)
    - If ranges are given (e.g., "4-32"), extract individual values if listed, otherwise include the range notation
    - Include abbreviated codes AND full descriptions when both are provided
    - Do not invent or assume options not explicitly shown
    - Combine information from all provided images to create a complete ordering scheme

    OUTPUT REQUIREMENTS:
    - position_options: List of lists of dictionaries, where each inner list contains the options for that position as key-value pairs where key is the option code and value is the description
    - position_descriptions: List describing what each position represents
    - manufacturer/component_type: If clearly identifiable

    EXAMPLE ANALYSIS:
    If you see a scheme like:
    Position 1 (Family): STM32 = Arm-based 32-bit microcontroller
    Position 2 (Series): F = general-purpose, L = low-power
    Position 3 (Pins): 32 = 32 pins, 48 = 48 pins, 64 = 64 pins

    Output: {
      "position_options": [
        [{"STM32": "Arm-based 32-bit microcontroller"}], 
        [{"F": "general-purpose"}, {"L": "low-power"}], 
        [{"32": "32 pins"}, {"48": "48 pins"}, {"64": "64 pins"}]
      ],
      "position_descriptions": ["Device Family", "Product Series", "Pin Count"]
    }

    Return ONLY the JSON object, no markdown formatting or extra text.

    Analyze the provided component datasheet images and extract the ordering scheme following the ComponentOrderingScheme model.
    """
    
    content = [prompt] + images
    response = model.generate_content(content)
    
    # Parse JSON response and return ComponentOrderingScheme
    import json
    print("Raw response:")
    print(response.text)
    print("=" * 50)
    
    try:
        # Strip markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()
        
        result_data = json.loads(response_text)
        return ComponentOrderingScheme(**result_data)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response text was:", repr(response.text))
        # Return a default structure for now
        return ComponentOrderingScheme(
            position_options=[],
            position_descriptions=[],
            manufacturer="Unknown",
            component_type="Unknown"
        )

async def extract_ordering_scheme_vlm(images: List[Image.Image]) -> ComponentOrderingScheme:
    
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    
    prompt = """
    You are analyzing component datasheets to extract ordering/part numbering schemes. These schemes define how manufacturers structure their part numbers with different options at each position.

    TASK OVERVIEW:
    Extract the complete part numbering structure from any ordering information, part number breakdown, or ordering code table you find in the images. The information may be spread across multiple images.

    ANALYSIS STEPS:
    1. IDENTIFY the ordering scheme section (often titled "Ordering Information", "Part Numbering", "Product Selection", etc.)
    2. LOCATE the hierarchical breakdown showing each position's meaning
    3. EXTRACT all possible options for each position in the part number
    4. CAPTURE position descriptions (e.g., "Package Type", "Memory Size", etc.)

    EXTRACTION RULES:
    - Include EVERY possible option code for each position
    - Maintain exact position order as shown in the source
    - Use exact text/codes from the document (preserve case, numbers, special characters)
    - If ranges are given (e.g., "4-32"), extract individual values if listed, otherwise include the range notation
    - Include abbreviated codes AND full descriptions when both are provided
    - Do not invent or assume options not explicitly shown
    - Combine information from all provided images to create a complete ordering scheme

    OUTPUT REQUIREMENTS:
    - position_options: List of lists of dictionaries, where each inner list contains the options for that position as key-value pairs where key is the option code and value is the description
    - position_descriptions: List describing what each position represents
    - manufacturer/component_type: If clearly identifiable

    EXAMPLE ANALYSIS:
    If you see a scheme like:
    Position 1 (Family): STM32 = Arm-based 32-bit microcontroller
    Position 2 (Series): F = general-purpose, L = low-power
    Position 3 (Pins): 32 = 32 pins, 48 = 48 pins, 64 = 64 pins

    Output: {
      "position_options": [
        [{"STM32": "Arm-based 32-bit microcontroller"}], 
        [{"F": "general-purpose"}, {"L": "low-power"}], 
        [{"32": "32 pins"}, {"48": "48 pins"}, {"64": "64 pins"}]
      ],
      "position_descriptions": ["Device Family", "Product Series", "Pin Count"]
    }

    Return ONLY the JSON object, no markdown formatting or extra text.

    Analyze the provided component datasheet images and extract the ordering scheme following the ComponentOrderingScheme model.
    """
    
    content = [prompt] + images
    # Wrap synchronous Gemini call in async executor
    import asyncio
    response = await asyncio.to_thread(model.generate_content, content)
    
    # Parse JSON response and return ComponentOrderingScheme
    import json
    print("Raw response:")
    print(response.text)
    print("=" * 50)
    
    try:
        # Strip markdown code blocks if present
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()
        
        result_data = json.loads(response_text)
        return ComponentOrderingScheme(**result_data)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response text was:", repr(response.text))
        # Return a default structure for now
        return ComponentOrderingScheme(
            position_options=[],
            position_descriptions=[],
            manufacturer="Unknown",
            component_type="Unknown"
        )

def process_single_pin_page_with_retry(image: Image.Image, page_num: int, first_result: dict, model, generation_config, max_retries: int = 3) -> Tuple[int, dict]:
    """Process a single pin table page with enhanced retry logic for API rate limits"""
    import time
    import random
    
    for attempt in range(max_retries):
        try:
            # Add exponential backoff with jitter for retries
            if attempt > 0:
                delay = (2 ** attempt) + random.uniform(0, 2)
                print(f"Image {page_num} retry {attempt + 1}/{max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)
            
            return process_single_pin_page(image, page_num, first_result, model, generation_config, max_retries=1)
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Image {page_num} failed after {max_retries} attempts: {e}")
                return page_num, {}
            else:
                print(f"Image {page_num} attempt {attempt + 1} failed: {e}. Retrying...")
                continue

def process_single_pin_page(image: Image.Image, page_num: int, first_result: dict, model, generation_config, max_retries: int = 1) -> Tuple[int, dict]:
    """Process a single pin table page with retry logic"""

    # Determine if this is simple or multi-package format
    is_simple_document = "pins" in first_result

    if is_simple_document:
        # Simple document format
        sample_pin = {}
        if first_result.get("pins"):
            sample_pin = next(iter(first_result["pins"].values()), {})

        field_names = list(sample_pin.keys()) if sample_pin else ["pin_name", "type", "function"]

        continuation_prompt = f"""
        Extract pin table data from page {page_num}. Use EXACT same SIMPLE structure as first page.

        Expected fields: {field_names}

        CRITICAL JSON RULES:
        1. Always escape quotes in values with backslash
        2. No trailing commas
        3. Use simple, short values - avoid complex descriptions
        4. IMPORTANT: Always use "pin_name" as the field for the actual pin/ball/signal name
        5. ALL VALUES MUST BE STRINGS - do not use nested objects or arrays
        6. If a field contains multiple values, join them with commas or semicolons

        Return valid JSON only, no markdown. If no pins found, return empty object {{}}.

        SIMPLE Structure: {{"pins": {{"PIN_NUMBER": {{"pin_name": "ACTUAL_PIN_NAME", "other_field": "value"}}}}}}
        """
    else:
        # Multi-package document format
        package_names = list(first_result.keys())

        # Find a non-empty package to get sample pin structure
        sample_pin = {}
        for package_data in first_result.values():
            if package_data:  # Check if package has pins
                sample_pin = next(iter(package_data.values()))
                break

        # If all packages are empty, use default field names
        if sample_pin:
            field_names = list(sample_pin.keys())
        else:
            # Fallback field names when all packages are empty
            field_names = ["pin_name", "type", "io_structure", "alternate_functions"]

        continuation_prompt = f"""
        Extract pin table data from page {page_num}. Use EXACT same MULTI-PACKAGE structure as first page.

        Expected packages: {package_names}
        Expected fields: {field_names}

        CRITICAL JSON RULES:
        1. Always escape quotes in values with backslash
        2. No trailing commas
        3. Use simple, short values - avoid complex descriptions
        4. If a field is long, truncate it
        5. Replace any problematic characters with underscores
        6. IMPORTANT: Always use "pin_name" as the field for the actual pin/ball/signal name
        7. ALL VALUES MUST BE STRINGS - do not use nested objects or arrays
        8. If a field contains multiple values, join them with commas or semicolons

        Return valid JSON only, no markdown. If no pins found, return empty object {{}}.

        MULTI-PACKAGE Structure: {{"PACKAGE_NAME": {{"PIN_NUMBER": {{"pin_name": "ACTUAL_PIN_NAME", "other_field": "value"}}}}}}
        """
    
    for attempt in range(max_retries + 1):
        try:
            # Use simple timeout without signals (thread-safe)
            response = model.generate_content([continuation_prompt, image], generation_config=generation_config)
            response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Fix common JSON issues before parsing
            response_text = response_text.replace('\n', ' ').replace('\r', '')
            
            # Check if JSON is complete
            if not response_text.endswith('}'):
                print(f"Image {page_num}: Truncated response, attempting to fix...")
                # Try to close JSON properly
                brace_count = response_text.count('{') - response_text.count('}')
                response_text += '}' * brace_count
            
            page_result = json.loads(response_text)
            
            # Clean up None values in the result
            cleaned_result = {}
            for package, pins in page_result.items():
                cleaned_pins = {}
                for pin_num, pin_data in pins.items():
                    cleaned_pin_data = {}
                    for field, value in pin_data.items():
                        # Convert nested objects to strings, None to empty string
                        if value is None:
                            cleaned_pin_data[field] = ""
                        elif isinstance(value, dict):
                            # Convert dict to string representation
                            cleaned_pin_data[field] = json.dumps(value)
                        elif isinstance(value, (list, tuple)):
                            # Convert list/tuple to string representation
                            cleaned_pin_data[field] = str(value)
                        else:
                            cleaned_pin_data[field] = str(value)
                    cleaned_pins[pin_num] = cleaned_pin_data
                cleaned_result[package] = cleaned_pins
            
            return page_num, cleaned_result
            
        except Exception as e:
            if attempt < max_retries:
                print(f"Image {page_num} attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(1)  # Brief delay before retry
            else:
                print(f"Image {page_num} failed after {max_retries + 1} attempts: {e}")
                return page_num, {}

def extract_pin_table_vlm(images: List[Image.Image], package_options: List[str], valid_permutations: List[str]) -> PinTableResult:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables")
        return PinTableResult(packages={})
    
    if len(api_key) < 30:
        print(f"WARNING: Google API key seems too short: {len(api_key)} chars")
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-pro")
    
    
    generation_config = genai.GenerationConfig(
        temperature=0.1,
        max_output_tokens=65536  # Increased to handle large pin tables
    )
    
    if not images:
        print("No images provided for pin table extraction")
        return PinTableResult(packages={})
    
    package_validation = f"Expected package types from ordering scheme: {package_options}" if package_options else ""
    valid_parts_info = f"Valid part numbers: {valid_permutations[:10]}..." if valid_permutations else ""
    
    # Process first image to establish structure
    print(f"Processing first image (1/{len(images)})...")
    first_prompt = f"""
    You are analyzing the first page of a semiconductor pin table from a datasheet. This page likely contains column headers and initial pin data.

    {package_validation}
    {valid_parts_info}

    ANALYSIS REQUIREMENTS:
    1. IDENTIFY all pin table data and column headers
    2. EXTRACT column structure and headers
    3. DETERMINE if this is a SIMPLE pin table (single package/device) or MULTI-PACKAGE pin table
    4. EXTRACT all pin data from this page

    DOCUMENT TYPE DETECTION:
    - SIMPLE DOCUMENT: Only one pin assignment per pin (no multiple package columns)
    - MULTI-PACKAGE DOCUMENT: Multiple package columns with different pin assignments per package

    EXTRACTION RULES:
    - Only extract data from actual pin tables
    - Package columns typically contain pin numbers (digits, letters like A1, B2, etc.)
    - Pin information columns contain pin names, types, functions, etc.
    - Empty cells or "-" mean the pin doesn't exist for that package
    - Use EXACT column names as found in the headers for package names
    - DO NOT create compound names like "STM32F722xx_LQFP64" - just use "LQFP64"

    OUTPUT FORMAT:

    FOR MULTI-PACKAGE DOCUMENTS:
    Return a dictionary where:
    - Keys are SIMPLE package names exactly matching column headers (e.g. "LQFP64", "UFBGA176")
    - Values are dictionaries mapping pin numbers to pin information
    - Example: {{"LQFP64": {{"1": {{"pin_name": "VDD", "function": "Power"}}}}}}

    FOR SIMPLE DOCUMENTS (no multiple packages OR no ordering info available):
    Return a dictionary with a single "pins" key:
    - Key is "pins"
    - Value is a dictionary mapping pin numbers directly to pin information
    - Example: {{"pins": {{"1": {{"pin_name": "VDD", "function": "Power"}}}}}}

    FIELD REQUIREMENTS:
    - IMPORTANT: Always use "pin_name" as the field for the actual pin/ball/signal name
    - Use short, clean field names for other columns
    - Avoid very long values - summarize if needed
    - ALL VALUES MUST BE STRINGS - do not use nested objects or arrays
    - If a field contains multiple values, join them with commas or semicolons

    Return ONLY valid JSON with no markdown formatting or extra text.
    """
    
    try:
        print("Sending first image to VLM...")
        print(f"Using Google API Key: {os.getenv('GOOGLE_API_KEY')[:20]}...{os.getenv('GOOGLE_API_KEY')[-4:]}")
        first_response = model.generate_content([first_prompt, images[0]], generation_config=generation_config)
        print("Received response from VLM")
        
        # Debug the full response object
        print(f"Response type: {type(first_response)}")
        print(f"Response attributes: {dir(first_response)}")
        
        if hasattr(first_response, 'text') and first_response.text:
            response_text = first_response.text.strip()
            print(f"Full response text length: {len(response_text)}")
        elif hasattr(first_response, 'candidates') and first_response.candidates:
            print(f"Found {len(first_response.candidates)} candidates")
            if first_response.candidates[0].content.parts:
                response_text = first_response.candidates[0].content.parts[0].text.strip()
                print(f"Got text from candidates: {len(response_text)} chars")
            else:
                print("No parts in candidate content")
                response_text = ""
        else:
            print("Warning: No text in Gemini response and no candidates")
            print(f"Response object: {first_response}")
            response_text = ""
        
        # Check for empty response and retry
        if not response_text or len(response_text) < 10:
            print("ERROR: Empty or very short response from Gemini API - this may indicate rate limiting")
            print("Retrying with delay...")
            import time
            time.sleep(5)  # Wait 5 seconds before retry
            
            try:
                retry_response = model.generate_content([first_prompt, images[0]], generation_config=generation_config)
                if hasattr(retry_response, 'text') and retry_response.text:
                    response_text = retry_response.text.strip()
                    print(f"Retry successful - response length: {len(response_text)}")
                else:
                    print("Retry also failed - proceeding with empty result")
                    return PinTableResult(packages={})
            except Exception as e:
                print(f"Retry failed with error: {e}")
                return PinTableResult(packages={})
        
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        print(f"Response: {response_text[:500]}...")
        
        if not response_text:
            print("ERROR: Empty response from Gemini API")
            return PinTableResult(packages={})
        
        if not response_text.endswith('}'):
            print("Warning: Response appears to be truncated")
            last_brace = response_text.rfind('}')
            if last_brace > 0:
                response_text = response_text[:last_brace + 1] + '}'
                print("Attempting to fix truncated JSON")
        
        print("JSON preview:", response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        try:
            first_result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text}")
            return PinTableResult(packages={})
        
        # Determine if this is a simple document or multi-package document
        is_simple_document = "pins" in first_result

        if is_simple_document:
            print(f"Detected SIMPLE document format")
            # Clean up None values in simple format
            if "pins" in first_result:
                cleaned_pins = {}
                for pin_num, pin_data in first_result["pins"].items():
                    cleaned_pin_data = {}
                    for field, value in pin_data.items():
                        # Convert nested objects to strings, None to empty string
                        if value is None:
                            cleaned_pin_data[field] = ""
                        elif isinstance(value, dict):
                            # Convert dict to string representation
                            cleaned_pin_data[field] = json.dumps(value)
                        elif isinstance(value, (list, tuple)):
                            # Convert list/tuple to string representation
                            cleaned_pin_data[field] = str(value)
                        else:
                            cleaned_pin_data[field] = str(value)
                    cleaned_pins[pin_num] = cleaned_pin_data
                first_result = {"pins": cleaned_pins}
                print(f"First image processed. Found {len(cleaned_pins)} pins in simple format")
        else:
            print(f"Detected MULTI-PACKAGE document format")
            # Clean up None values in multi-package format
            cleaned_first_result = {}
            for package, pins in first_result.items():
                cleaned_pins = {}
                for pin_num, pin_data in pins.items():
                    cleaned_pin_data = {}
                    for field, value in pin_data.items():
                        # Convert nested objects to strings, None to empty string
                        if value is None:
                            cleaned_pin_data[field] = ""
                        elif isinstance(value, dict):
                            # Convert dict to string representation
                            cleaned_pin_data[field] = json.dumps(value)
                        elif isinstance(value, (list, tuple)):
                            # Convert list/tuple to string representation
                            cleaned_pin_data[field] = str(value)
                        else:
                            cleaned_pin_data[field] = str(value)
                    cleaned_pins[pin_num] = cleaned_pin_data
                cleaned_first_result[package] = cleaned_pins

            first_result = cleaned_first_result
            print(f"First image processed. Found packages: {list(first_result.keys())}")
        
    except Exception as e:
        import traceback
        print(f"ERROR processing first image: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Full traceback: {traceback.format_exc()}")
        try:
            if 'first_response' in locals() and hasattr(first_response, 'text'):
                print(f"Response preview: {first_response.text[:200]}")
            elif 'first_response' in locals():
                print(f"Response object: {first_response}")
        except:
            print("Could not access response object")
        return PinTableResult(packages={})
    

    combined_result = first_result.copy()
    if len(images) > 1:
        print(f"Processing remaining {len(images) - 1} images with rate limiting...")
        
        # Process remaining images with controlled concurrency to avoid API rate limits
        import asyncio
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Use large thread pool for long-running API calls (~1min each)
        with ThreadPoolExecutor(max_workers=50) as executor:
            remaining_images = [(images[i], i + 1) for i in range(1, len(images))]
            
            # Submit tasks with staggered delays to avoid rate limits
            future_to_page = {}
            for idx, (image, page_num) in enumerate(remaining_images):
                # Minimal delay for burst submission of long-running requests
                if idx > 0:
                    time.sleep(0.01)  # 10ms delay - minimal since requests take ~1min each
                
                future = executor.submit(
                    process_single_pin_page_with_retry, 
                    image, page_num, first_result, model, generation_config
                )
                future_to_page[future] = page_num
            
            # Collect results as they complete
            for future in as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    page_num_result, page_result = future.result()
                    
                    if is_simple_document:
                        # Simple document: merge pins directly
                        if "pins" in page_result and "pins" in combined_result:
                            combined_result["pins"].update(page_result["pins"])
                        elif "pins" in page_result:
                            combined_result["pins"] = page_result["pins"]

                        pins_found = len(page_result.get("pins", {}))
                        print(f"Image {page_num_result} processed. Found {pins_found} new pins.")
                    else:
                        # Multi-package document: merge by package
                        for package, pins in page_result.items():
                            if package in combined_result:
                                combined_result[package].update(pins)
                            else:
                                combined_result[package] = pins

                        pins_found = sum(len(pins) for pins in page_result.values())
                        print(f"Image {page_num_result} processed. Found {pins_found} new pins.")
                    
                except Exception as e:
                    import traceback
                    print(f"Unexpected error processing image {page_num}: {type(e).__name__}: {str(e)}")
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
    
    if is_simple_document:
        total_pins = len(combined_result.get("pins", {}))
        print(f"Pin table extraction complete. Total pins extracted: {total_pins}")
        return SimplePinTableResult(pins=combined_result.get("pins", {}))
    else:
        total_pins = sum(len(pins) for pins in combined_result.values())
        print(f"Pin table extraction complete. Total pins extracted: {total_pins}")
        return PinTableResult(packages=combined_result)

def map_permutation_to_pin_table(pin_table_packages: List[str], permutations: List[str], ordering_screenshots: List[Image.Image]) -> Dict[str, List[List[str]]]:
    """
    Analyze each pin table package against ordering scheme to determine which permutation 
    options are valid for each specific package.
    """
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-pro")
    
    generation_config = genai.GenerationConfig(
        temperature=0.1,
        max_output_tokens=16384
    )
    
    prompt = f"""
    Map each pin table package to its constrained ordering scheme options.
    
    PIN TABLE PACKAGES: {pin_table_packages}
    SAMPLE PERMUTATIONS: {permutations[:20]}...
    
    CRITICAL: The sample permutations show the COMPLETE structure. Count the positions in the permutations to determine the exact number of positions needed.
    
    ANALYSIS TASK:
    1. Study the ordering information to understand the COMPLETE hierarchical structure
    2. For each pin table package, create a full constrained ordering scheme
    3. Include ALL positions from the original ordering scheme, not just the constrained ones
    
    MAPPING LOGIC:
    - Each pin table package constrains specific positions (e.g., package type, pin count)
    - For constrained positions: include only the options that match this package
    - For unconstrained positions: include ALL original options from the ordering scheme
    - The result must have the SAME number of positions as the sample permutations
    
    OUTPUT FORMAT:
    Return a JSON object where:
    - Keys are pin table package names  
    - Values are arrays of arrays (COMPLETE ordering scheme for that package)
    - Each value must have the SAME length as the sample permutations
    - Include every position, whether constrained or not
    
    EXAMPLE: If permutations are 8 characters long, each package must have exactly 8 position arrays.
    
    Return ONLY the JSON object, no markdown formatting or explanations.
    """
    
    try:
        content = [prompt] + ordering_screenshots
        response = model.generate_content(content, generation_config=generation_config)
        
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        package_constraints = json.loads(response_text)
        print(f"Package-specific constraints: {package_constraints}")
        
        return package_constraints
        
    except Exception as e:
        print(f"Error mapping pin packages to ordering constraints: {e}")
        return {}


def score_table_for_pins(table_data) -> int:
    """Score a table based on how likely it is to be a PIN TABLE specifically."""
    try:
        # Basic dimension check - pin tables should have reasonable size
        if table_data.shape[0] < 3 or table_data.shape[1] < 3:
            return 0

        # Get headers from column names (since we already used first row as headers)
        headers = [str(h).lower().strip() for h in table_data.columns]
        score = 0

        # STRONG indicators of pin tables (must have at least one)
        pin_table_indicators = ["pin", "ball", "terminal", "pad"]
        has_pin_indicator = any(indicator in ' '.join(headers) for indicator in pin_table_indicators)

        if not has_pin_indicator:
            # If no clear pin indicators in headers, check first column data
            if table_data.shape[0] > 0:
                first_col_data = [str(cell).strip().upper() for cell in table_data.iloc[:, 0] if str(cell).strip()][:5]
                pin_data_count = sum(1 for cell in first_col_data
                                   if (cell.isdigit() or
                                       re.match(r'^[A-Z]\d+$', cell) or
                                       cell in ['VDD', 'VSS', 'GND', 'VCC', 'NC', 'AVDD', 'DVDD']))
                if pin_data_count < 2:  # Need at least 2 pin-like entries
                    return 0

        # Score based on header content
        pin_keywords = ["pin", "ball", "terminal", "pad"]
        function_keywords = ["function", "signal", "name", "description", "type", "alternate", "af"]
        package_keywords = ["lqfp", "bga", "qfn", "wlcsp", "package"]

        for header in headers:
            # Strong pin table indicators
            for keyword in pin_keywords:
                if keyword in header:
                    score += 50  # Much higher weight

            # Function/signal columns (typical in pin tables)
            for keyword in function_keywords:
                if keyword in header:
                    score += 25

            # Package-specific columns
            for keyword in package_keywords:
                if keyword in header:
                    score += 30

        # Score based on data content - look for pin-like patterns
        if table_data.shape[0] > 0:
            first_col_data = [str(cell).strip() for cell in table_data.iloc[:, 0] if str(cell).strip()][:10]
            pin_like_count = 0

            for cell in first_col_data:
                if cell.isdigit() and 1 <= int(cell) <= 300:  # Reasonable pin numbers
                    pin_like_count += 1
                elif re.match(r'^[A-Z]\d+$', cell.upper()):  # BGA style (A1, B2, etc.)
                    pin_like_count += 1
                elif cell.upper() in ['VDD', 'VSS', 'GND', 'VCC', 'NC', 'AVDD', 'DVDD', 'VDDA', 'VSSA']:
                    pin_like_count += 1

            # Need significant pin-like content
            pin_ratio = pin_like_count / min(len(first_col_data), 10)
            if pin_ratio >= 0.3:  # At least 30% pin-like
                score += pin_like_count * 15
            else:
                score -= 50  # Penalize tables without pin-like data

        # HEAVILY penalize non-pin table types
        bad_keywords = ["min", "max", "typical", "units", "conditions", "parameter", "voltage", "current",
                       "frequency", "temperature", "specification", "characteristics", "symbol", "test"]
        bad_count = sum(1 for h in headers for kw in bad_keywords if kw in h)
        if bad_count >= 2:
            score -= 100  # Heavy penalty for electrical specs tables

        # Penalize if looks like feature/overview tables
        overview_keywords = ["feature", "overview", "description", "benefits", "applications"]
        if any(kw in ' '.join(headers) for kw in overview_keywords):
            score -= 50

        # Pin tables should have multiple packages/columns
        if table_data.shape[1] >= 5:  # Pin tables often have many package columns
            score += 20
        elif table_data.shape[1] < 4:
            score -= 20

        # Pin tables should have decent number of rows (pins)
        if table_data.shape[0] >= 10:
            score += 30
        elif table_data.shape[0] < 5:
            score -= 30

        return max(0, score)  # Ensure non-negative score

    except Exception as e:
        print(f"Error in score_table_for_pins: {e}")
        return 0

def quick_scan_pdfplumber(pdf_path: Path, page_num: int) -> int:
    """Quick scan of a page using pdfplumber - just get the best score"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num <= len(pdf.pages):
                page = pdf.pages[page_num - 1]
                tables = page.extract_tables()

                best_score = 0
                for table in tables:
                    if table and len(table) > 1 and len(table[0]) >= 2:
                        # Quick DataFrame creation
                        import pandas as pd
                        headers = [f'Col_{j}' if h is None or h == '' else str(h).strip()
                                 for j, h in enumerate(table[0])]
                        df = pd.DataFrame(table[1:], columns=headers).fillna('')

                        if df.shape[0] >= 1 and df.shape[1] >= 2:
                            score = score_table_for_pins(df)
                            best_score = max(best_score, score)

                return best_score
    except Exception:
        return 0

def process_pin_descriptions(pdf_path: Path, start_page: int, end_page: int, fallback_mode: bool = False):
    """Simple pin table page range detection - just find the continuous range"""

    page_scores = {}

    # Limit page range to prevent excessive processing (max 50 pages)
    end_page = min(end_page, start_page + 50)
    print(f"🔍 Scanning pages {start_page} to {end_page} for pin tables...")

    # Adjust threshold based on mode
    if fallback_mode:
        threshold = 30  # Lower threshold for fallback mode (full document scan)
        print(f"🔧 Using fallback mode with lower threshold ({threshold})")
    else:
        threshold = 50  # Higher threshold for TOC-based detection

    # Quick scan to find pages with table-like content
    for page_num in range(start_page, end_page + 1):
        page_score = 0

        # Use pdfplumber for table detection (no Ghostscript concurrency issues)
        page_score = quick_scan_pdfplumber(pdf_path, page_num)

        if page_score > threshold:
            page_scores[page_num] = page_score
            print(f"✅ Page {page_num}: pin table candidate (score: {page_score})")
        else:
            print(f"⚪ Page {page_num}: no pin table detected (score: {page_score})")

    if not page_scores:
        print("❌ No pin tables found in the specified page range")
        return None

    # Find the longest continuous sequence of pages with pin tables
    print(f"\n📊 ANALYSIS: Found {len(page_scores)} pages with potential pin tables")

    # Sort pages by page number
    sorted_pages = sorted(page_scores.keys())

    # Find the longest continuous sequence
    best_start = sorted_pages[0]
    best_end = sorted_pages[0]
    current_start = sorted_pages[0]
    current_end = sorted_pages[0]

    for i in range(1, len(sorted_pages)):
        page = sorted_pages[i]
        prev_page = sorted_pages[i-1]

        if page == prev_page + 1:
            # Continuous sequence continues
            current_end = page
        else:
            # Sequence breaks - check if current sequence is the longest
            if (current_end - current_start) > (best_end - best_start):
                best_start = current_start
                best_end = current_end

            # Start new sequence
            current_start = page
            current_end = page

    # Check final sequence
    if (current_end - current_start) > (best_end - best_start):
        best_start = current_start
        best_end = current_end

    print(f"🎯 Pin table detected: pages {best_start} to {best_end} ({best_end - best_start + 1} pages)")

    # Show summary of what was found
    total_score = sum(page_scores[p] for p in range(best_start, best_end + 1))
    avg_score = total_score / (best_end - best_start + 1)
    print(f"📈 Average pin table score: {avg_score:.1f}")

    screenshots = take_screenshots(pdf_path, best_start, best_end)
    return screenshots

def extract_pin_table_and_ordering_info_sync(pdf_path: str) -> Dict[str, Any]:
    """Synchronous version for thread pool execution"""
    with PDF_SEMAPHORE:  # Allow limited concurrent PDF operations
        try:
            return _extract_pin_table_and_ordering_info_impl(pdf_path)
        except Exception as e:
            return {"error": f"Failed to process PDF: {str(e)}"}

def _extract_pin_table_and_ordering_info_impl(pdf_path: str) -> Dict[str, Any]:
    """Common implementation for both sync and async versions"""
    pdf_path_obj = Path(pdf_path)
    
    # Extract table of contents
    toc = extract_toc(pdf_path_obj)

    # Check PDF page count for fallback logic
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    doc.close()

    # Try to find relevant sections from TOC if available
    sections = None
    use_fallback = False

    if not toc:
        use_fallback = True
        print(f"No TOC found.")
    else:
        # Find relevant sections from TOC
        sections = find_relevant_sections_sync(toc)

        # Check if sections are useful - trigger fallback if sections are too narrow or missing
        pin_range = 0
        ordering_range = 0

        if hasattr(sections, 'pin_descriptions') and sections.pin_descriptions:
            pin_range = sections.pin_descriptions.end_page - sections.pin_descriptions.start_page + 1

        if hasattr(sections, 'ordering_information') and sections.ordering_information:
            ordering_range = sections.ordering_information.end_page - sections.ordering_information.start_page + 1

        # Trigger fallback if sections are too narrow (< 3 pages total coverage) or missing
        total_section_coverage = pin_range + ordering_range
        print(f"🔍 Section analysis: pin_range={pin_range}, ordering_range={ordering_range}, total={total_section_coverage}")
        print(f"🔍 Sections found: pin_descriptions={'✅' if sections.pin_descriptions else '❌'}, ordering_information={'✅' if sections.ordering_information else '❌'}")

        # Be more aggressive with fallback for small PDFs
        should_fallback = (
            total_section_coverage < 5 or  # Increased threshold from 3 to 5
            (not sections.pin_descriptions and not sections.ordering_information) or
            (pin_range <= 2 and total_pages < 30)  # Small pin range in small PDF
        )

        if should_fallback:
            use_fallback = True
            print(f"TOC sections too narrow or missing (pin: {pin_range}, ordering: {ordering_range} pages, total_pages: {total_pages}). Using fallback.")
        else:
            print(f"TOC sections adequate (pin: {pin_range}, ordering: {ordering_range} pages). Using TOC-based extraction.")

    if use_fallback:
        # Fallback: if PDF is small (<50 pages), scan all pages
        if total_pages < 50:
            print(f"PDF has {total_pages} pages (<50). Using fallback: scanning all pages.")
            # Create fake sections covering the entire document
            from types import SimpleNamespace
            sections = SimpleNamespace(
                pin_descriptions=SimpleNamespace(start_page=1, end_page=total_pages),
                ordering_information=SimpleNamespace(start_page=1, end_page=total_pages)
            )
        else:
            return {"error": f"Could not extract useful sections from {total_pages}-page PDF (too large for full scan)"}
    
    result = {
        "pin_table": [],
        "ordering_info": {},
        "sections_found": sections
    }
    
    # Process ordering information if found
    if sections.ordering_information:
        start_page, end_page = sections.ordering_information.start_page, sections.ordering_information.end_page
        try:
            ordering_result = process_ordering_information_sync(pdf_path_obj, start_page, end_page)
            if ordering_result:
                result["ordering_info"] = ordering_result
        except Exception as e:
            print(f"Warning: Could not process ordering information: {e}")
    
    # Process pin descriptions if found
    if sections.pin_descriptions:
        start_page, end_page = sections.pin_descriptions.start_page, sections.pin_descriptions.end_page
        try:
            # Check if we're in fallback mode (scanning full document)
            is_fallback = (start_page == 1 and end_page == total_pages)
            pin_screenshots = process_pin_descriptions(pdf_path_obj, start_page, end_page, fallback_mode=is_fallback)
            if pin_screenshots:
                # Get package options from ordering info if available
                package_options = []
                valid_permutations = []
                if result.get("ordering_info") and result["ordering_info"].get("position_options"):
                    for pos_group in result["ordering_info"]["position_options"]:
                        for option in pos_group:
                            if "value" in option:
                                package_options.append(option["value"])
                
                # Pin table screenshots prepared for VLM processing
                print(f"Prepared {len(pin_screenshots)} pin table pages for VLM extraction")
                
                pin_table_result = extract_pin_table_vlm(pin_screenshots, package_options, valid_permutations)

                # Handle both simple and multi-package results
                if isinstance(pin_table_result, SimplePinTableResult):
                    result["pin_table"] = pin_table_result.pins
                elif isinstance(pin_table_result, PinTableResult):
                    result["pin_table"] = pin_table_result.packages
                else:
                    result["pin_table"] = {}
        except Exception as e:
            result["pin_table"] = {}
            print(f"Warning: Could not process pin descriptions: {e}")
    
    return result

async def extract_pin_table_and_ordering_info(pdf_path: str) -> Dict[str, Any]:
    """Main async function to extract pin table and ordering information from a PDF"""
    try:
        pdf_path_obj = Path(pdf_path)
        
        # Extract table of contents
        toc = extract_toc(pdf_path_obj)
        if not toc:
            return {"error": "Could not extract table of contents"}
        
        # Find relevant sections (async version)
        sections = await find_relevant_sections(toc)
        
        result = {
            "pin_table": [],
            "ordering_info": {},
            "sections_found": sections
        }
        
        # Process ordering information if found
        if sections.ordering_information:
            start_page, end_page = sections.ordering_information.start_page, sections.ordering_information.end_page
            try:
                ordering_result = await process_ordering_information(pdf_path_obj, start_page, end_page)
                if ordering_result:
                    result["ordering_info"] = ordering_result
            except Exception as e:
                print(f"Warning: Could not process ordering information: {e}")
        
        # Process pin descriptions if found
        if sections.pin_descriptions:
            start_page, end_page = sections.pin_descriptions.start_page, sections.pin_descriptions.end_page
            try:
                # Check if we're in fallback mode (scanning full document)
                is_fallback = (start_page == 1 and end_page == total_pages)
                pin_screenshots = process_pin_descriptions(pdf_path_obj, start_page, end_page, fallback_mode=is_fallback)
                if pin_screenshots:
                    # Get package options from ordering info if available
                    package_options = []
                    valid_permutations = []
                    if result.get("ordering_info") and result["ordering_info"].get("position_options"):
                        for pos_group in result["ordering_info"]["position_options"]:
                            for option in pos_group:
                                if "value" in option:
                                    package_options.append(option["value"])
                    
                    # Pin table screenshots prepared for VLM processing
                    print(f"Prepared {len(pin_screenshots)} pin table pages for VLM extraction")
                    
                    pin_table_result = extract_pin_table_vlm(pin_screenshots, package_options, valid_permutations)
                    result["pin_table"] = pin_table_result.packages if pin_table_result else {}
            except Exception as e:
                result["pin_table"] = {}
                print(f"Warning: Could not process pin descriptions: {e}")
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to process PDF: {str(e)}"}
