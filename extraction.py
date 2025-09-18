"""
Standalone PDF rule extraction service for extracting rules from PDF documents.

This service:
1. Connects directly to MongoDB (no Django required)
2. Extracts and orders chunks by page number from chunk database
3. Groups chunks by exact title paths with precise page ranges
4. Splits page ranges >10 pages into 5-page groups with 1-page overlap
5. Sends chunks with RULE_EXTRACTION_PROMPT to LLM with pin table
6. Structures LLM output into required JSON format
7. Saves structured rules to test_rules.json file
"""

import json
import os
import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


async def retry_with_backoff(
    func, 
    *args, 
    max_retries: int = 3, 
    backoff_seconds: float = 10.0,
    **kwargs
):
    """
    Retry function with fixed backoff for OpenAI API rate limits.
    
    Args:
        func: The async function to retry
        *args: Arguments for the function
        max_retries: Maximum number of retry attempts
        backoff_seconds: Fixed backoff time in seconds (default 60s for OpenAI rate limits)
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call
    
    Raises:
        The last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                logger.error(f"All {max_retries + 1} attempts failed. Final error: {str(e)}")
                raise e
            
            # Check if it's a rate limit error (common patterns)
            error_str = str(e).lower()
            if any(term in error_str for term in ['rate limit', 'quota', 'too many requests', '429']):
                logger.warning(f"Rate limit hit on attempt {attempt + 1}/{max_retries + 1}. Retrying in {backoff_seconds}s...")
                await asyncio.sleep(backoff_seconds)
            else:
                # For other errors, use exponential backoff with jitter
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Error on attempt {attempt + 1}/{max_retries + 1}: {str(e)}. Retrying in {sleep_time:.1f}s...")
                await asyncio.sleep(sleep_time)
    
    raise last_exception


RULE_EXTRACTION_PROMPT = """
**Task: Extract Hierarchical Schematic-Design Rules from {device_name} Section {section_title}**

You are an *expert electrical-design analyst*. Your only job is: read this section of {device_name} datasheet and produce a **clean, machine-readable JSON list of schematic-level design rules**.

### FORMATTING RULES

* Each rule = **exactly one sentence** with complete technical specification
* Include all **numerical values, units, tolerances, ranges, operating conditions**
* Use **exact pin/signal names** as in the document
* State operating modes, voltages, frequencies if applicable
* Rules must be **precise and measurable** on a schematic diagram file or its corresponding netlist
* **CRITICAL: Rules must be completely standalone - do NOT reference figures, tables, or other document sections** (e.g., don't say "in Fig. 5" or "as shown in Table 15")
* Include all necessary technical details directly in the rule text rather than referencing external content

### PIN EXTRACTION

* Always include a `"pin_names"` array listing every pin name mentioned in the rule
* Pin names must match datasheet text exactly (e.g., `"VDD"`, `"VDDA"`, `"RESET"`, `"PA0"`)
* Keep pin ranges as-is (`PA[7:0]` → `["PA[7:0]"]`)
* For supply rules, include **all supply pins** mentioned
* For interface rules, include **all related signal pins**
* If the rule is not attached to specific pins, use `"pins": []`

### CONDITIONS EXTRACTION

* Always include a `"conditions"` array listing specific schematic conditions that must be met for the rule to apply
* Use simple, single-topic focused natural language strings
* Examples: `["ADC is enabled"]`, `["External crystal is used"]`, `["High-speed mode is selected"]`
* Most rules are unconditional (apply generally) - use `"conditions": []` for these
* Only add conditions when the rule explicitly states prerequisites or specific operating modes or schematic design configurations where it applies.

### PACKAGES EXTRACTION

* Always include a `"packages"` array listing specific device packages that the rule applies to
* Use exact package names as mentioned in the datasheet (e.g., `["PackageType1"]`, `["PackageType2"]`)
* If the rule applies to all packages or no specific package is mentioned, use `"packages": []`
* Examples: `["Package1", "Package2"]` for rules specific to certain package variants
* Only add packages when the rule explicitly states it applies to specific package types

### ESSENTIAL CLASSIFICATION

Each rule must have an `"essential"` boolean:

* `"essential": true` → absolutely critical for device power-up & operation:
  * Power supply ranges, ground connections
  * Required external components (crystal, decoupling capacitors)
  * Critical pull-ups/downs, safety-critical specs

* `"essential": false` → recommended but not mandatory:
  * Performance optimization, best practices, thermal/layout guidance

### SCHEMATIC VERIFIABILITY

Include **only rules verifiable in a schematic/netlist**:

* ✅ Power pins, supply values, external passives, pin interconnects, component values
* ❌ Exclude rules needing oscilloscope/thermal/EMI/EMC/timing lab tests

### CATEGORY REQUIREMENTS

Group rules by **specific technical categories**, e.g.:

* `"Power Supply Requirements"`
* `"Decoupling Capacitor Requirements"`
* `"Crystal Oscillator Requirements"`
* `"GPIO Electrical Characteristics"`
* `"Reset Circuit Requirements"`
* `"ADC Reference Voltage"`

### EXAMPLE OUTPUT FORMAT

Return a JSON array with this structure:

```json
[
  {{
    "rule": "Supply VDD with voltage between 1.8V and 3.6V at maximum current of 50mA",
    "category": "Power Supply Requirements", 
    "essential": true,
    "pins": ["VDD"],
    "packages":[],
    "conditions": []
  }},
  {{
    "rule": "Connect 100nF ceramic decoupling capacitor between VDD and VSS within 5mm of device",
    "category": "Decoupling Capacitor Requirements",
    "essential": true, 
    "pins": ["VDD", "VSS"],
    "packages": [],
    "conditions": []
  }},
  {{
    "rule": "Use 4.7kΩ pull-up resistor on RESET pin to VDD for proper reset operation",
    "category": "Reset Circuit Requirements",
    "essential": true,
    "pins": ["RESET", "VDD"],
    "packages": [],
    "conditions": []
  }},
  {{
    "rule": "Connect 32.768kHz crystal between OSC32_IN and OSC32_OUT pins with 12pF load capacitors to ground",
    "category": "Crystal Oscillator Requirements",
    "essential": true,
    "pins": ["OSC32_IN", "OSC32_OUT"],
    "packages": [],
    "conditions": ["Low-speed external crystal is used", "RTC functionality is required"]
  }}
]
```

**Section Context:**
- Section: {section_title}
- Title path: {title_path}
- Pages: {page_refs}

**Pin Context:**
{pin_context}

**Content to analyze:**
{combined_content}

**Note:** If this section does not contain schematic-level design content, return `[]` only.
"""

@dataclass
class PageRange:
    """Represents a page range for a section."""
    start: int
    end: int
    section_title: str
    level: int

@dataclass 
class ChunkGroup:
    """Represents a group of chunks for processing."""
    chunks: List[Dict[str, Any]]
    page_range: PageRange
    section_title: str


class ChecklistItem(BaseModel):
    rule: str
    category: str
    essential: bool
    pins: List[str]
    packages: List[str]
    conditions: List[str]


class RulesChecklist(BaseModel):
    checklist: List[ChecklistItem]


class StandalonePDFRuleExtractor:
    """Standalone service for extracting rules from PDF documents."""
    
    def __init__(
        self, 
        organization_name: str, 
        mongodb_uri: str = None,
        max_workers: int = 100,
        max_retries: int = 3,
        retry_backoff_seconds: float = 60.0
    ):
        self.organization_name = organization_name
        self.mongodb_uri = mongodb_uri or os.getenv("MONGODB_URI")
        self.chunk_collection_name = f"{organization_name}_chunks"
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        
        # Don't create semaphore here - will be created in correct event loop
        self.semaphore = None
        
        if not self.mongodb_uri:
            raise ValueError("MongoDB URI must be provided via parameter or MONGODB_URI environment variable")
        
        # Configure MongoDB client with connection pooling for high concurrency
        self.client = MongoClient(
            self.mongodb_uri,
            maxPoolSize=max_workers + 10,  # Pool size larger than worker count
            minPoolSize=10,                # Minimum connections to maintain
            maxIdleTimeMS=30000,          # Close idle connections after 30s
            waitQueueTimeoutMS=10000,     # Wait up to 10s for connection
            serverSelectionTimeoutMS=5000, # Server selection timeout
            socketTimeoutMS=20000,        # Socket timeout for operations
            connectTimeoutMS=10000,       # Connection timeout
            retryWrites=True,             # Enable retry writes
            retryReads=True,              # Enable retry reads
            w='majority',                 # Write concern
            readPreference='secondaryPreferred'  # Prefer secondary for reads
        )
        self.db = self.client["voltai-backend"]
        self.collection = self.db[self.chunk_collection_name]
        
        logger.info(f"Connected to MongoDB collection: {self.chunk_collection_name}")
        logger.info(f"Configured with {max_workers} max workers, {max_retries} max retries, {retry_backoff_seconds}s backoff")
        logger.info(f"MongoDB pool: maxPoolSize={max_workers + 10}, minPoolSize=10")
    
    async def extract_rules_from_pdf(self, pdf_name: str, pin_table_data: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Main method to extract rules from a PDF document.
        
        Args:
            pdf_name: Name of the PDF file to extract rules from
            pin_table_data: Optional pin table data to include in extraction
            
        Returns:
            List of extracted rules in the specified JSON format
        """
        try:
            # Create semaphore in the current event loop if not already created
            if self.semaphore is None:
                self.semaphore = asyncio.Semaphore(self.max_workers)
            
            # Step 1: Extract and order chunks by page number
            chunks = await self._get_ordered_chunks(pdf_name)
            if not chunks:
                logger.info(f"No chunks found for PDF {pdf_name}")
                return []
            
            # Step 2: Group chunks by title_path
            section_groups = await self._group_chunks_by_sections(chunks)
            
            # Step 3: Split large page ranges into smaller groups
            processed_groups = await self._split_large_page_ranges(section_groups)
            
            # Step 4: Extract rules from each group in parallel
            print(f"Starting parallel extraction for {len(processed_groups)} groups with {self.max_workers} max workers...")
            
            # Create tasks for parallel execution
            tasks = []
            for i, group in enumerate(processed_groups, 1):
                task = self._extract_rules_from_group_with_semaphore(group, pin_table_data, pdf_name, i, len(processed_groups))
                tasks.append(task)
            
            # Run all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            all_rules = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to extract rules from group {i+1}: {str(result)}")
                else:
                    all_rules.extend(result)
            
            # Step 5: Save to test_rules.json
            await self._save_rules_to_file(all_rules)
            
            return all_rules
            
        except Exception as e:
            logger.error(f"Error extracting rules from PDF {pdf_name}: {str(e)}")
            return []
    
    async def _get_ordered_chunks(self, pdf_name: str) -> List[Dict[str, Any]]:
        """Extract chunks from database and order by page number."""
        try:
            # Query chunks from MongoDB
            chunks = []
            
            # Remove .pdf extension and create regex pattern to match filename starts with
            pdf_base_name = pdf_name.replace('.pdf', '').replace('.PDF', '')
            filename_pattern = {"$regex": f"^{pdf_base_name}", "$options": "i"}
            
            logger.info(f"Searching MongoDB for filename pattern: {filename_pattern} (base name: {pdf_base_name})")
            
            # Find all chunks where filename starts with the PDF base name  
            chunk_count = 0
            cursor = self.collection.find({"data.filename": filename_pattern}).max_time_ms(30000).limit(10000).batch_size(100)
            for chunk in cursor:
                chunk_count += 1
                data = chunk.get("data", {})
                raw_page = data.get("page", 0)
                if isinstance(raw_page, list) and raw_page:
                    page_num = min([p for p in raw_page if isinstance(p, int)] or [0])
                elif isinstance(raw_page, int):
                    page_num = raw_page
                else:
                    page_num = 0

                # Extract bbox coordinates for proper ordering
                bbox_info = data.get("bbox", [])
                bbox_coords = None
                if bbox_info and isinstance(bbox_info, list) and len(bbox_info) > 0:
                    first_bbox = bbox_info[0]
                    if isinstance(first_bbox, dict) and "coordinates" in first_bbox:
                        coords = first_bbox["coordinates"]
                        if len(coords) >= 4:
                            # coordinates format: [x1, y1, x2, y2]
                            bbox_coords = coords

                chunk_data = {
                    "key": chunk.get("key"),
                    "content": data.get("llm_context", ""),
                    "page": page_num,
                    "filename": data.get("filename", ""),
                    "type": data.get("type", "text"),
                    "title": data.get("title", ""),
                    "title_path": data.get("title_path", []),
                    "bbox": bbox_coords,
                    "metadata": data,
                }
                chunks.append(chunk_data)
            
            # Sort by page number first, then by bbox coordinates (top-to-bottom, left-to-right)
            chunks.sort(key=self._get_chunk_sort_key)
            
            logger.info(f"Found {chunk_count} chunks for PDF {pdf_name} (pattern: {pdf_base_name}*)")
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting ordered chunks: {str(e)}")
            return []
    
    def _get_chunk_sort_key(self, chunk: Dict[str, Any]) -> tuple:
        """Generate sort key for chunks: page first, then bbox coordinates for within-page ordering"""
        page = chunk.get("page", 0)
        if not isinstance(page, int):
            page = 0
        
        bbox = chunk.get("bbox")
        if bbox and len(bbox) >= 4:
            # coordinates format: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox[:4]
            # Sort by: page, then top-to-bottom (y1), then left-to-right (x1)
            return (page, y1, x1)
        else:
            # No bbox available, just use page and a default position
            return (page, 0, 0)
    
    async def _group_chunks_by_sections(self, chunks: List[Dict[str, Any]]) -> List[ChunkGroup]:
        """Group chunks by title_path-derived sections with precise page ranges across all pages."""
        try:
            if not chunks:
                return []

            # Group by title_path (using only first 2 elements to reduce group count)
            groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
            for chunk in chunks:
                title_path = chunk.get("title_path") or []
                # Use only first 2 elements of title_path
                truncated_path = title_path[:2] if title_path else []
                key: Tuple[str, ...] = tuple(truncated_path) if truncated_path else ("Document",)
                groups[key].append(chunk)

            section_groups: List[ChunkGroup] = []
            for key, section_chunks in groups.items():
                if not section_chunks:
                    continue
                start_page = min(c.get("page", 0) for c in section_chunks)
                end_page = max(c.get("page", 0) for c in section_chunks)
                if key == ("Document",):
                    section_title = "Document"
                    level = 1
                else:
                    section_title = key[-1]
                    level = len(key)
                page_range = PageRange(
                    start=start_page,
                    end=end_page,
                    section_title=section_title,
                    level=level,
                )
                section_groups.append(ChunkGroup(
                    chunks=sorted(section_chunks, key=lambda c: c.get("page", 0)),
                    page_range=page_range,
                    section_title=section_title,
                ))

            # Sort groups by starting page for stable downstream processing
            section_groups.sort(key=lambda g: (g.page_range.start, g.page_range.level))

            # Log coverage info
            all_pages = [c.get("page", 0) for c in chunks if isinstance(c.get("page", 0), int)]
            if all_pages:
                logger.info(
                    f"Created {len(section_groups)} title_path groups covering pages {min(all_pages)}-{max(all_pages)}"
                )
            else:
                logger.info(f"Created {len(section_groups)} title_path groups")
            return section_groups
            
        except Exception as e:
            logger.error(f"Error grouping chunks by sections: {str(e)}")
            return []
    
    async def _split_large_page_ranges(self, section_groups: List[ChunkGroup]) -> List[ChunkGroup]:
        """Split page ranges >10 pages into 5-page groups with 1-page overlap."""
        try:
            processed_groups = []
            
            for group in section_groups:
                page_range = group.page_range
                range_size = page_range.end - page_range.start + 1
                
                if range_size <= 10:
                    # Keep as is
                    processed_groups.append(group)
                else:
                    # Split into 5-page groups with 1-page overlap
                    current_start = page_range.start
                    group_index = 1
                    
                    while current_start <= page_range.end:
                        current_end = min(current_start + 4, page_range.end)  # 5 pages (0-4 = 5)
                        
                        # Get chunks in this subrange
                        subrange_chunks = [
                            chunk for chunk in group.chunks 
                            if current_start <= chunk["page"] <= current_end
                        ]
                        
                        if subrange_chunks:
                            new_page_range = PageRange(
                                start=current_start,
                                end=current_end,
                                section_title=f"{page_range.section_title} (Part {group_index})",
                                level=page_range.level
                            )
                            
                            processed_groups.append(ChunkGroup(
                                chunks=subrange_chunks,
                                page_range=new_page_range,
                                section_title=f"{group.section_title} (Part {group_index})"
                            ))
                        
                        # Move to next group with 1-page overlap
                        current_start += 4  # Next start overlaps by 1 page
                        group_index += 1
            
            logger.info(f"Split into {len(processed_groups)} processed groups")
            return processed_groups
            
        except Exception as e:
            logger.error(f"Error splitting large page ranges: {str(e)}")
            return section_groups
    
    async def _extract_rules_from_group_with_semaphore(self, group: ChunkGroup, pin_table_data: Optional[str], pdf_name: str, group_index: int, total_groups: int) -> List[Dict[str, Any]]:
        """Extract rules from a chunk group using semaphore for rate limiting."""
        async with self.semaphore:
            print(f"[{group_index}/{total_groups}] Extracting rules from group: {group.section_title}")
            return await retry_with_backoff(
                self._extract_rules_from_group,
                group, pin_table_data, pdf_name,
                max_retries=self.max_retries,
                backoff_seconds=self.retry_backoff_seconds
            )
    
    async def _extract_rules_from_group(self, group: ChunkGroup, pin_table_data: Optional[str], pdf_name: str) -> List[Dict[str, Any]]:
        """Extract rules from a chunk group using LLM."""
        try:
            # Build prompt context
            device_name = os.path.splitext(os.path.basename(pdf_name))[0]
            section_title = group.section_title
            title_path_list = group.chunks[0].get("title_path") or []
            title_path = " -> ".join(title_path_list) if title_path_list else section_title
            page_refs_list = sorted({c.get("page", 0) for c in group.chunks if isinstance(c.get("page", 0), int)})
            page_refs = ", ".join(str(p) for p in page_refs_list) if page_refs_list else ""
            
            logger.info(f"Processing section '{section_title}' with {len(group.chunks)} chunks on pages {page_refs}")
            
            pin_context = f"Pin context:\n{pin_table_data}\n" if pin_table_data else ""
            combined_content = "\n\n".join([
                f"[Page {chunk['page']}]\n{chunk['content']}"
                for chunk in group.chunks
            ])
            
            content_length = len(combined_content)
            logger.info(f"Prepared prompt with {content_length} characters for section '{section_title}'")

            # Prepare prompt
            prompt = RULE_EXTRACTION_PROMPT.format(
                device_name=device_name,
                section_title=section_title,
                title_path=title_path,
                page_refs=page_refs,
                pin_context=pin_context,
                combined_content=combined_content,
            )
            
            # Get LLM response using the async method
            response = await self._call_llm_async(prompt, section_title)
            
            # Parse structured/JSON response
            try:
                if hasattr(response, "model_dump"):
                    rules_data = response.model_dump()
                    logger.debug(f"Model dump for '{section_title}': {rules_data}")
                    rules = rules_data.get("checklist", [])
                elif isinstance(response, dict):
                    rules = response.get("checklist", [])
                elif isinstance(response, str):
                    rules = json.loads(response).get("checklist", [])
                else:
                    logger.warning(f"Unexpected LLM response type for section '{group.section_title}': {type(response)}")
                    logger.warning(f"Response content: {response}")
                    return []
                
                # Add section information to each rule
                for rule in rules:
                    if isinstance(rule, dict):
                        rule['section'] = section_title
                        rule['title_path'] = title_path
                
                logger.info(f"Extracted {len(rules)} rules from section '{section_title}'")
                return rules
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response for section {group.section_title}: {str(e)}")
                logger.error(f"Problematic response: {response}")
                return []
            
        except Exception as e:
            logger.error(f"Error extracting rules from group {group.section_title}: {type(e).__name__}: {str(e)}")
            logger.error(f"Full error details: {repr(e)}")
            # Print the full traceback for debugging
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    async def _call_llm_async(self, prompt: str, section_title: str):
        """Call LLM with structured output in proper async context."""
        try:
            # Create LLM client in the current async context to avoid event loop binding issues
            system_prompt = (
                "Extract concise, actionable design rules from the provided text. "
                "Return items with fields: rule, category, essential, pins, conditions."
            )

            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-5",
                temperature=0.0,
                max_tokens=16000,
                max_retries=2,
            )
            
            # llm = ChatAnthropic(
            #     api_key=os.getenv("ANTHROPIC_API_KEY"),
            #     model="claude-sonnet-4-20250514",
            #     temperature=0.0,
            #     max_tokens=16000,
            #     max_retries=2,
            # )
            structured_llm = llm.with_structured_output(RulesChecklist)
            chat_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}"),
            ])
            chain = chat_prompt | structured_llm
            
            # Track tokens if token_tracker is available
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(self, 'token_tracker') and self.token_tracker:
                # Count tokens in input
                full_prompt = system_prompt + prompt
                input_tokens = self.token_tracker.count_tokens(full_prompt)
            
            logger.info(f"Calling LLM for section '{section_title}' (estimated input tokens: {input_tokens})...")
            response = await chain.ainvoke({"input": prompt})
            logger.info(f"LLM response received for section '{section_title}' - type: {type(response)}")
            
            if hasattr(self, 'token_tracker') and self.token_tracker:
                # Count tokens in output
                if hasattr(response, 'model_dump'):
                    response_str = str(response.model_dump())
                else:
                    response_str = str(response)
                output_tokens = self.token_tracker.count_tokens(response_str)
                
                # Add to total tracking
                self.token_tracker.add_usage(input_tokens, output_tokens)
                logger.info(f"Token usage for '{section_title}': {input_tokens} input, {output_tokens} output")
            
            # Debug: Log the raw response
            logger.info(f"Raw LLM response for '{section_title}': {response}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error calling LLM for sectionca '{section_title}': {type(e).__name__}: {str(e)}")
            raise
    
    async def _save_rules_to_file(self, rules: List[Dict[str, Any]]) -> None:
        """Save structured rules to test_rules.json file."""
        try:
            output_data = {"checklist": rules}
            
            with open("test_rules.json", "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(rules)} rules to test_rules.json")
            
        except Exception as e:
            logger.error(f"Error saving rules to file: {str(e)}")
    
    def cleanup(self):
        """Cleanup MongoDB connections"""
        try:
            if hasattr(self, 'client') and self.client:
                self.client.close()
                logger.info("MongoDB client connections closed")
        except Exception as e:
            logger.error(f"Error closing MongoDB connections: {str(e)}")


async def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract rules from PDF documents")
    parser.add_argument("organization", help="Organization name")
    parser.add_argument("pdf_name", help="PDF filename to extract rules from")
    parser.add_argument("--pin-table", help="Optional pin table data file path")
    parser.add_argument("--output", default="test_rules.json", help="Output file path (default: test_rules.json)")
    parser.add_argument("--mongodb-uri", help="MongoDB URI (or set MONGODB_URI env var)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load pin table data if provided
    pin_table_data = None
    if args.pin_table:
        try:
            with open(args.pin_table, 'r', encoding='utf-8') as f:
                pin_table_data = f.read()
        except Exception as e:
            logger.error(f"Error reading pin table file {args.pin_table}: {e}")
            return
    
    # Initialize extractor
    try:
        extractor = StandalonePDFRuleExtractor(args.organization, args.mongodb_uri)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Extract rules
    try:
        rules = await extractor.extract_rules_from_pdf(args.pdf_name, pin_table_data)
        
        # Save to custom output file if specified
        if args.output != "test_rules.json":
            output_data = {"checklist": rules}
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(rules)} rules to {args.output}")
        
        print(f"Successfully extracted {len(rules)} rules from {args.pdf_name}")
        
    except Exception as e:
        logger.error(f"Failed to extract rules: {e}")
        return


if __name__ == "__main__":
    asyncio.run(main())