#!/usr/bin/env python
"""
Minimal Supabase Standalone Script
This script provides similar functionality to minimal_django_script.py but uses pure Supabase interface
instead of Django models.
"""

import json
import os
import sys
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm

# Add the backend directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from supabase import create_client, Client
except ImportError:
    print("Error: Required packages not found. Please install:")
    print("pip install supabase python-dotenv tqdm")
    sys.exit(1)


class SupabaseConstants:
    URL = "https://zwyjsybljgykghsixwus.supabase.co"
    KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp3eWpzeWJsamd5a2doc2l4d3VzIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTY4Njk0Njk1MiwiZXhwIjoyMDAyNTIyOTUyfQ.Lnq-NZh_L6zCfVa2i_6SiT-eIiRmQaBhxwxEmk2OBY8"
    ACTIVITIES = "activities"
    ROLES = "roles"
    USERS = "user_user"
    CHATS = "conversation_chat"
    CONVERSATIONS = "conversation_conversation"
    FEEDBACKS = "feedback_feedback"
    ORGANIZATIONS = "organization_organization"


# Initialize Supabase client
def get_supabase_client() -> Client:
    """Initialize and return Supabase client"""
    if not SupabaseConstants.URL or not SupabaseConstants.KEY:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set")
        sys.exit(1)
    
    return create_client(SupabaseConstants.URL, SupabaseConstants.KEY)

# Initialize client
supabase: Client = get_supabase_client()

@dataclass
class PartData:
    """Data class for Part information"""
    part_id: str
    pin_table: Dict = None
    created_at: str = None
    updated_at: str = None

@dataclass
class ChecklistData:
    """Data class for Checklist information"""
    uuid: str
    name: str
    part_id: str = None
    project_id: str = None
    is_generated: bool = True
    is_public: bool = True
    is_deleted: bool = False
    created_at: str = None
    updated_at: str = None

@dataclass
class RuleData:
    """Data class for Rule information"""
    uuid: str
    checklist_id: str
    content: str
    category: str = "Default"
    level: str = "RECOMMENDED"  # ESSENTIAL or RECOMMENDED
    pins: List = None
    is_deleted: bool = False
    created_at: str = None
    updated_at: str = None

@dataclass
class OrganizationData:
    """Data class for Organization information"""
    uuid: str
    name: str
    slug: str
    is_active: bool = True
    created_at: str = None
    updated_at: str = None

@dataclass
class TopicData:
    """Data class for Topic information"""
    uuid: str
    name: str
    organization_id: str
    is_active: bool = True
    created_at: str = None
    updated_at: str = None

@dataclass
class SourceData:
    """Data class for Source information"""
    uuid: str
    name: str
    filename: str
    topic_id: str
    organization_id: str
    file_type: str = "pdf"
    is_active: bool = True
    metadata: Dict = None
    created_at: str = None
    updated_at: str = None

def clean_json_data(obj):
    """Recursively clean JSON data by removing null bytes and invalid Unicode characters"""
    if isinstance(obj, dict):
        return {k: clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(item) for item in obj]
    elif isinstance(obj, str):
        # Remove null bytes and other problematic characters
        # Replace \u0000 with empty string or a space
        cleaned = obj.replace('\u0000', '')
        # Remove other control characters (0x00-0x1F) except tab, newline, carriage return
        return cleaned
    else:
        return obj

def format_pin_table(pin_data, footnote=None):
    """Format pin data and footnote into a single JSON structure"""
    # The pin_table should contain both pin information and footnote
    result = {
        "pins": pin_data,
    }
    if footnote:
        result["footnote"] = footnote
    
    # Clean the data before returning
    return clean_json_data(result)

def get_part_by_id(part_id: str) -> Optional[Dict]:
    """Get a part by its ID from Supabase"""
    try:
        result = supabase.table("schematic_part").select("*").eq("part_id", part_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error fetching part {part_id}: {str(e)}")
        return None

def create_or_update_part(part_id: str, pin_table: Dict) -> Tuple[bool, Dict]:
    """Create or update a part in Supabase"""
    try:
        # Check if part exists
        existing_part = get_part_by_id(part_id)
        
        part_data = {
            "part_id": part_id,
            "pin_table": pin_table
        }
        
        if existing_part:
            # Update existing part
            result = supabase.table("schematic_part").update(part_data).eq("part_id", part_id).execute()
            return False, result.data[0] if result.data else {}
        else:
            # Create new part
            result = supabase.table("schematic_part").insert(part_data).execute()
            return True, result.data[0] if result.data else {}
            
    except Exception as e:
        print(f"Error creating/updating part {part_id}: {str(e)}")
        raise

def get_checklist_by_name_and_part(name: str, part_id: str) -> Optional[Dict]:
    """Get a checklist by name and part ID"""
    try:
        result = supabase.table("schematic_checklist").select("*").eq("name", name).eq("part_id", part_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error fetching checklist {name}: {str(e)}")
        return None

def create_checklist(name: str, part_id: str, is_generated: bool = True, is_public: bool = True) -> Dict:
    """Create a new checklist in Supabase"""
    try:
        checklist_data = {
            "uuid": str(uuid.uuid4()),
            "name": name,
            "part_id": part_id,
            "is_generated": is_generated,
            "is_public": is_public,
            "is_deleted": False
        }
        
        result = supabase.table("schematic_checklist").insert(checklist_data).execute()
        return result.data[0] if result.data else {}
        
    except Exception as e:
        print(f"Error creating checklist {name}: {str(e)}")
        raise

def create_rule(checklist_id: str, content: str, category: str = "Default", level: str = "RECOMMENDED", pins: List = None) -> Dict:
    """Create a new rule in Supabase"""
    try:
        rule_data = {
            "uuid": str(uuid.uuid4()),
            "checklist_id": checklist_id,
            "content": content,
            "category": category,
            "level": level,
            "pins": pins or [],
            "is_deleted": False
        }
        
        result = supabase.table("schematic_rule").insert(rule_data).execute()
        return result.data[0] if result.data else {}
        
    except Exception as e:
        print(f"Error creating rule: {str(e)}")
        raise

def bulk_insert_parts(parts_data: List[Dict]) -> int:
    """Bulk insert parts into Supabase"""
    try:
        result = supabase.table("schematic_part").insert(parts_data).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"Error bulk inserting parts: {str(e)}")
        raise

def bulk_insert_checklists(checklists_data: List[Dict]) -> int:
    """Bulk insert checklists into Supabase"""
    try:
        result = supabase.table("schematic_checklist").insert(checklists_data).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"Error bulk inserting checklists: {str(e)}")
        raise

def bulk_insert_rules(rules_data: List[Dict]) -> int:
    """Bulk insert rules into Supabase"""
    try:
        result = supabase.table("schematic_rule").insert(rules_data).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"Error bulk inserting rules: {str(e)}")
        raise

def populate_parts_from_json(json_file_path: str) -> Tuple[int, int, int]:
    """Populate Part, Checklist, and Rule tables from a JSON file"""
    
    with open(json_file_path, "r") as f:
        data = json.load(f)
    
    created_parts = 0
    created_checklists = 0
    created_rules = 0
    failed_parts = []
    
    # Process each part individually to handle errors gracefully
    for part_id, part_data in tqdm(data.items()):
        try:
            # Create or update Part
            pin_table_data = format_pin_table(
                part_data.get("pin", []),
                part_data.get("footnote")
            )
            
            part_created, part_info = create_or_update_part(part_id, pin_table_data)
            
            if part_created:
                created_parts += 1
                print(f"Created part: {part_id}")
            else:
                print(f"Updated part: {part_id}")
            
            # Process checklists for this part
            checklist_items = part_data.get("checklist", [])
            
            if checklist_items:
                # Create a checklist for this part
                checklist_name = part_id
                existing_checklist = get_checklist_by_name_and_part(checklist_name, part_id)
                
                if not existing_checklist:
                    checklist_info = create_checklist(checklist_name, part_id)
                    created_checklists += 1
                    print(f"  Created checklist: {checklist_name}")
                    
                    # Create rules for each checklist item
                    for checklist_item in checklist_items:
                        category = checklist_item.get("category", "Default")
                        items = checklist_item.get("items", [])
                        
                        for item_content in items:
                            # Clean the rule content before saving
                            cleaned_content = clean_json_data(item_content)
                            cleaned_category = clean_json_data(category)
                            
                            create_rule(
                                checklist_info["uuid"],
                                cleaned_content,
                                cleaned_category
                            )
                            created_rules += 1
                            print(f"    Created rule in category '{cleaned_category}': {cleaned_content[:50]}...")
                else:
                    print(f"  Checklist already exists, skipping: {checklist_name}")
                            
        except Exception as e:
            print(f"\nERROR processing part {part_id}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            failed_parts.append({
                'part_id': part_id,
                'error': str(e),
                'error_type': type(e).__name__
            })
            continue
    
    # Report any failures
    if failed_parts:
        print(f"\n{'='*50}")
        print(f"FAILED PARTS ({len(failed_parts)} total):")
        print(f"{'='*50}")
        for failure in failed_parts:
            print(f"Part ID: {failure['part_id']}")
            print(f"  Error: {failure['error']}")
            print(f"  Type: {failure['error_type']}")
    
    return created_parts, created_checklists, created_rules

def populate_parts_from_json_bulk_optimized(json_file_path: str, batch_size: int = 1000, use_ignore_conflicts: bool = True) -> Tuple[int, int, int]:
    """
    Optimized version using bulk operations for better performance on large datasets.
    
    Args:
        json_file_path: Path to the JSON file containing part data
        batch_size: Number of records to process in each batch (default: 1000)
        use_ignore_conflicts: If True, handles conflicts gracefully (default: True)
    """
    
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Clean up part IDs by removing unwanted tags
    tags_to_clear = ['Internal code', 'Blank', 'No character']
    print(f"Cleaning part IDs by removing tags: {tags_to_clear}")
    cleaned_data = {}
    
    cleaned_part_id = os.path.basename(json_file_path).split(".")[0].upper()
    
    for _, part_data in data.items():
        cleaned_data[cleaned_part_id] = part_data
    
    # Update data to use cleaned part IDs
    data = cleaned_data
    print(f"Part ID cleaning complete. Now have {len(data)} parts.")
    
    created_parts = 0
    created_checklists = 0
    created_rules = 0
    
    part_id = cleaned_part_id
    print(f"Part ID: {part_id}")
    
    # Check if part already exists
    existing_part = get_part_by_id(part_id)
    
    # Prepare Part objects
    print(f"Preparing {len(data)} parts...")
    parts_to_create = []
    
    for _, part_data in tqdm(data.items(), desc="Preparing parts"):
        pin_table_data = format_pin_table(
            part_data.get("pin", []),
            part_data.get("footnote")
        )
        
        if not existing_part:
            parts_to_create.append({
                "part_id": part_id,
                "pin_table": pin_table_data
            })
    
    # Bulk create parts
    if parts_to_create:
        print("\nBulk creating parts...")
        for i in tqdm(range(0, len(parts_to_create), batch_size), desc="Creating parts"):
            batch = parts_to_create[i:i + batch_size]
            created_count = bulk_insert_parts(batch)
            created_parts += created_count
    
    # Prepare checklists
    print("\nPreparing checklists...")
    checklists_to_create = []
    checklist_uuid_mapping = {}
    
    for part_id, part_data in data.items():
        if part_data.get("checklist"):
            checklist_name = part_id
            
            # Check if checklist already exists
            existing_checklist = get_checklist_by_name_and_part(checklist_name, part_id)
            
            if not existing_checklist:
                checklist_uuid = str(uuid.uuid4())
                checklists_to_create.append({
                    "uuid": checklist_uuid,
                    "name": checklist_name,
                    "part_id": part_id,
                    "is_generated": True,
                    "is_public": True,
                    "is_deleted": False
                })
                checklist_uuid_mapping[checklist_uuid] = (checklist_name, part_id)
    
    # Bulk create checklists
    if checklists_to_create:
        print(f"\nCreating {len(checklists_to_create)} checklists...")
        for i in tqdm(range(0, len(checklists_to_create), batch_size), desc="Creating checklists"):
            batch = checklists_to_create[i:i + batch_size]
            created_count = bulk_insert_checklists(batch)
            created_checklists += created_count
    
    # Prepare rules in bulk
    print("\nPreparing rules...")
    rules_to_create = []
    
    for checklist_uuid, (checklist_name, part_id) in checklist_uuid_mapping.items():
        part_data = data.get(part_id, {})
        
        for checklist_item in part_data.get("checklist", []):
            category = clean_json_data(checklist_item.get("category", "Default"))
            item_content = checklist_item.get("rule", "")
            
            if item_content:
                rules_to_create.append({
                    "uuid": str(uuid.uuid4()),
                    "checklist_id": checklist_uuid,
                    "content": clean_json_data(item_content.strip()),
                    "category": category,
                    "level": "ESSENTIAL" if checklist_item.get("essential") else "RECOMMENDED",
                    "pins": checklist_item.get("pins", []),
                    "is_deleted": False
                })
    
    # Bulk create rules
    if rules_to_create:
        print(f"\nCreating {len(rules_to_create)} rules...")
        for i in tqdm(range(0, len(rules_to_create), batch_size), desc="Creating rules"):
            batch = rules_to_create[i:i + batch_size]
            created_count = bulk_insert_rules(batch)
            created_rules += created_count
    
    print(f"\n{'='*50}")
    print(f"OPTIMIZED BULK OPERATION COMPLETED:")
    print(f"{'='*50}")
    print(f"Parts created: {created_parts}")
    print(f"Checklists created: {created_checklists}")
    print(f"Rules created: {created_rules}")
    
    return created_parts, created_checklists, created_rules

def examine_db() -> bool:
    """
    Validate the integrity of Part, Checklist, and Rule tables.
    
    Performs the following checks:
    1. Check there are no two parts that have the same part_id
    2. Check that no two checklists have the same name
    3. For all the checklists linked to a part, check the checklist name is equal to the part id
    4. For rules, check that for all rules within the same checklist and under the same category,
       there are no duplicate contents
    """
    print("=" * 80)
    print("DATABASE VALIDATION REPORT")
    print("=" * 80)
    
    validation_passed = True
    
    try:
        # 1. Check for duplicate part_ids
        print("\n1. Checking for duplicate part_ids...")
        parts_result = supabase.table("schematic_part").select("part_id").execute()
        part_ids = [row["part_id"] for row in parts_result.data] if parts_result.data else []
        unique_part_ids = set(part_ids)
        
        if len(part_ids) != len(unique_part_ids):
            validation_passed = False
            print("   ❌ FAILED: Found duplicate part_ids!")
            
            from collections import Counter
            part_id_counts = Counter(part_ids)
            duplicates = {k: v for k, v in part_id_counts.items() if v > 1}
            
            for part_id, count in duplicates.items():
                print(f"      - part_id '{part_id}' appears {count} times")
        else:
            print(f"   ✓ PASSED: All {len(part_ids)} part_ids are unique")
        
        # 2. Check for duplicate checklist names
        print("\n2. Checking for duplicate checklist names...")
        checklists_result = supabase.table("schematic_checklist").select("name, uuid, part_id").execute()
        checklist_names = [row["name"] for row in checklists_result.data] if checklists_result.data else []
        unique_checklist_names = set(checklist_names)
        
        if len(checklist_names) != len(unique_checklist_names):
            validation_passed = False
            print("   ❌ FAILED: Found duplicate checklist names!")
            
            from collections import Counter
            checklist_name_counts = Counter(checklist_names)
            duplicates = {k: v for k, v in checklist_name_counts.items() if v > 1}
            
            for name, count in duplicates.items():
                print(f"      - checklist name '{name}' appears {count} times")
                duplicate_checklists = [row for row in checklists_result.data if row["name"] == name]
                for checklist in duplicate_checklists:
                    print(f"        - Checklist UUID: {checklist['uuid']}, Part: {checklist.get('part_id', 'None')}")
        else:
            print(f"   ✓ PASSED: All {len(checklist_names)} checklist names are unique")
        
        # 3. Check checklist-part relationships
        print("\n3. Checking checklist-part relationships...")
        checklists_with_parts = [row for row in checklists_result.data if row.get("part_id")] if checklists_result.data else []
        
        issues_found = False
        part_checklist_mapping = {}
        
        for checklist in checklists_with_parts:
            # Check if checklist name equals part_id
            if checklist["name"] != checklist["part_id"]:
                validation_passed = False
                issues_found = True
                print(f"   ❌ FAILED: Checklist name '{checklist['name']}' does not match part_id '{checklist['part_id']}'")
                print(f"      - Checklist UUID: {checklist['uuid']}")
            
            # Track parts with multiple checklists
            part_id = checklist["part_id"]
            if part_id not in part_checklist_mapping:
                part_checklist_mapping[part_id] = []
            part_checklist_mapping[part_id].append(checklist)
        
        # Check for parts with multiple checklists
        parts_with_multiple_checklists = {k: v for k, v in part_checklist_mapping.items() if len(v) > 1}
        
        if parts_with_multiple_checklists:
            validation_passed = False
            issues_found = True
            print("\n   ❌ FAILED: Found parts with multiple checklists!")
            for part_id, checklists in parts_with_multiple_checklists.items():
                print(f"      - Part '{part_id}' has {len(checklists)} checklists:")
                for checklist in checklists:
                    print(f"        - Checklist UUID: {checklist['uuid']}, Name: '{checklist['name']}'")
        
        if not issues_found:
            print(f"   ✓ PASSED: All checklist-part relationships are valid")
            print(f"      - {len(checklists_with_parts)} checklists are linked to parts")
            print(f"      - All checklist names match their part_ids")
            print(f"      - No parts have multiple checklists")
        
        # 4. Check for duplicate rules within same checklist and category
        print("\n4. Checking for duplicate rules within same checklist and category...")
        
        rules_result = supabase.table("schematic_rule").select("*").eq("is_deleted", False).execute()
        rules = rules_result.data if rules_result.data else []
        
        duplicate_rules_found = False
        total_rules_checked = len(rules)
        
        # Group rules by checklist
        rules_by_checklist = {}
        for rule in rules:
            checklist_id = rule["checklist_id"]
            if checklist_id not in rules_by_checklist:
                rules_by_checklist[checklist_id] = []
            rules_by_checklist[checklist_id].append(rule)
        
        for checklist_id, checklist_rules in rules_by_checklist.items():
            # Group rules by category
            rules_by_category = {}
            for rule in checklist_rules:
                category = rule.get("category", "Default")
                if category not in rules_by_category:
                    rules_by_category[category] = []
                rules_by_category[category].append(rule)
            
            # Check for duplicates within each category
            for category, category_rules in rules_by_category.items():
                # Track content and rules
                content_to_rules = {}
                for rule in category_rules:
                    content = rule["content"].strip()  # Normalize by stripping whitespace
                    if content not in content_to_rules:
                        content_to_rules[content] = []
                    content_to_rules[content].append(rule)
                
                # Find duplicates
                duplicates = {k: v for k, v in content_to_rules.items() if len(v) > 1}
                
                if duplicates:
                    validation_passed = False
                    duplicate_rules_found = True
                    
                    # Get checklist name for reporting
                    checklist_info = next((c for c in checklists_result.data if c["uuid"] == checklist_id), {})
                    checklist_name = checklist_info.get("name", "Unknown")
                    
                    print(f"\n   ❌ FAILED: Found duplicate rules in checklist '{checklist_name}' (UUID: {checklist_id})")
                    print(f"      Category: '{category}'")
                    
                    for content, duplicate_rules_list in duplicates.items():
                        print(f"      - Content appears {len(duplicate_rules_list)} times: '{content[:100]}{'...' if len(content) > 100 else ''}'")
                        for rule in duplicate_rules_list:
                            print(f"        - Rule UUID: {rule['uuid']}")
        
        if not duplicate_rules_found:
            print(f"   ✓ PASSED: No duplicate rules found")
            print(f"      - Checked {total_rules_checked} rules across {len(rules_by_checklist)} checklists")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        validation_passed = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if validation_passed:
        print("✓ All validation checks PASSED!")
    else:
        print("❌ Some validation checks FAILED - please review the issues above")
    
    # Additional statistics
    try:
        parts_count = len(supabase.table("schematic_part").select("part_id").execute().data or [])
        checklists_count = len(supabase.table("schematic_checklist").select("uuid").execute().data or [])
        rules_count = len(supabase.table("schematic_rule").select("uuid").execute().data or [])
        checklists_with_parts_count = len([c for c in checklists_result.data if c.get("part_id")] if checklists_result.data else [])
        checklists_without_parts_count = len([c for c in checklists_result.data if not c.get("part_id")] if checklists_result.data else [])
        
        print("\nDatabase Statistics:")
        print(f"  - Total Parts: {parts_count}")
        print(f"  - Total Checklists: {checklists_count}")
        print(f"  - Total Rules: {rules_count}")
        print(f"  - Checklists with Parts: {checklists_with_parts_count}")
        print(f"  - Checklists without Parts: {checklists_without_parts_count}")
    except Exception as e:
        print(f"Error fetching statistics: {str(e)}")
    
    return validation_passed

def create_checklist_with_rules(title: str, rules: List[Dict]):
    """Create a standalone checklist with rules (no part association)"""
    try:
        # Create checklist
        checklist_data = {
            "uuid": str(uuid.uuid4()),
            "name": title,
            "is_public": True,
            "is_generated": True,
            "is_deleted": False
        }
        
        checklist_result = supabase.table("schematic_checklist").insert(checklist_data).execute()
        checklist = checklist_result.data[0] if checklist_result.data else {}
        checklist_id = checklist.get("uuid")
        
        if not checklist_id:
            raise Exception("Failed to create checklist")
        
        # Create rules
        rules_to_create = []
        for rule_group in rules:
            category = rule_group.get('category', 'Default')
            all_rules = rule_group.get('items', [])
            
            for rule_content in all_rules:
                rules_to_create.append({
                    "uuid": str(uuid.uuid4()),
                    "checklist_id": checklist_id,
                    "content": rule_content,
                    "category": category,
                    "level": "RECOMMENDED",
                    "pins": [],
                    "is_deleted": False
                })
        
        # Bulk insert rules
        if rules_to_create:
            bulk_insert_rules(rules_to_create)
            print(f"Created {len(rules_to_create)} rules")
        
        print(f"Successfully created checklist '{title}' with {len(rules_to_create)} rules")
        
    except Exception as e:
        print(f"Error creating checklist with rules: {str(e)}")
        raise

# Organization Management Functions
def get_organization_by_name(name: str) -> Optional[Dict]:
    """Get an organization by its name from Supabase"""
    try:
        result = supabase.table("organization_organization").select("*").eq("name", name).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error fetching organization {name}: {str(e)}")
        return None

def get_organization_by_slug(slug: str) -> Optional[Dict]:
    """Get an organization by its slug from Supabase"""
    try:
        result = supabase.table("organization_organization").select("*").eq("slug", slug).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error fetching organization {slug}: {str(e)}")
        return None

def create_organization(name: str, slug: str = None) -> Dict:
    """Create a new organization in Supabase"""
    try:
        org_slug = slug or name.lower().replace(" ", "-")
        
        organization_data = {
            "uuid": str(uuid.uuid4()),
            "name": name,
            "slug": org_slug,
            "is_active": True
        }
        
        result = supabase.table("organization_organization").insert(organization_data).execute()
        return result.data[0] if result.data else {}
        
    except Exception as e:
        print(f"Error creating organization {name}: {str(e)}")
        raise

# Topic Management Functions
def get_topic_by_name_and_org(name: str, organization_id: str) -> Optional[Dict]:
    """Get a topic by name and organization ID"""
    try:
        result = supabase.table("source_topic").select("*").eq("name", name).eq("organization_id", organization_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error fetching topic {name}: {str(e)}")
        return None

def get_topics_by_organization(organization_id: str) -> List[Dict]:
    """Get all topics for an organization"""
    try:
        result = supabase.table("source_topic").select("*").eq("organization_id", organization_id).eq("is_active", True).execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"Error fetching topics for organization {organization_id}: {str(e)}")
        return []

def create_topic(name: str, organization_id: str) -> Dict:
    """Create a new topic in Supabase"""
    try:
        topic_data = {
            "uuid": str(uuid.uuid4()),
            "name": name,
            "organization_id": organization_id,
            "is_active": True
        }
        
        result = supabase.table("source_topic").insert(topic_data).execute()
        return result.data[0] if result.data else {}
        
    except Exception as e:
        print(f"Error creating topic {name}: {str(e)}")
        raise

# Source Management Functions
def get_source_by_filename_and_topic(filename: str, topic_id: str) -> Optional[Dict]:
    """Get a source by filename and topic ID"""
    try:
        result = supabase.table("source_source").select("*").eq("filename", filename).eq("topic_id", topic_id).execute()
        if result.data:
            return result.data[0]
        return None
    except Exception as e:
        print(f"Error fetching source {filename}: {str(e)}")
        return None

def get_sources_by_topic(topic_id: str) -> List[Dict]:
    """Get all sources for a topic"""
    try:
        result = supabase.table("source_source").select("*").eq("topic_id", topic_id).eq("is_active", True).execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"Error fetching sources for topic {topic_id}: {str(e)}")
        return []

def get_sources_by_topic_and_organization(topic_name: str, organization_name: str) -> List[Dict]:
    """Get all sources of a certain topic in an organization - optimized single query"""
    try:
        # Use a join query to get sources with topic and organization info in one call
        result = supabase.table("source_source").select("""
            *,
            source_topic!inner(name, organization_id),
            source_topic.organization_organization!inner(name, slug)
        """).eq("source_topic.name", topic_name).eq("source_topic.organization_organization.name", organization_name).eq("is_active", True).execute()
        
        return result.data if result.data else []
    except Exception as e:
        print(f"Error fetching sources for topic '{topic_name}' in organization '{organization_name}': {str(e)}")
        return []

def create_source(name: str, filename: str, topic_id: str, organization_id: str, file_type: str = "pdf", metadata: Dict = None) -> Dict:
    """Create a new source in Supabase"""
    try:
        source_data = {
            "uuid": str(uuid.uuid4()),
            "name": name,
            "filename": filename,
            "topic_id": topic_id,
            "organization_id": organization_id,
            "file_type": file_type,
            "is_active": True,
            "metadata": metadata or {}
        }
        
        result = supabase.table("source_source").insert(source_data).execute()
        return result.data[0] if result.data else {}
        
    except Exception as e:
        print(f"Error creating source {name}: {str(e)}")
        raise

def bulk_insert_sources(sources_data: List[Dict]) -> int:
    """Bulk insert sources into Supabase"""
    try:
        result = supabase.table("source_source").insert(sources_data).execute()
        return len(result.data) if result.data else 0
    except Exception as e:
        print(f"Error bulk inserting sources: {str(e)}")
        raise

# Utility Functions for Organization/Topic/Source Management
def setup_organization_structure(org_name: str, topics: List[str]) -> Tuple[Dict, List[Dict]]:
    """Set up an organization with topics"""
    try:
        # Create or get organization
        existing_org = get_organization_by_name(org_name)
        if existing_org:
            print(f"Organization '{org_name}' already exists")
            organization = existing_org
        else:
            organization = create_organization(org_name)
            print(f"Created organization: {org_name}")
        
        org_id = organization["uuid"]
        
        # Create topics
        created_topics = []
        for topic_name in topics:
            existing_topic = get_topic_by_name_and_org(topic_name, org_id)
            if existing_topic:
                print(f"Topic '{topic_name}' already exists")
                created_topics.append(existing_topic)
            else:
                topic = create_topic(topic_name, org_id)
                created_topics.append(topic)
                print(f"Created topic: {topic_name}")
        
        return organization, created_topics
        
    except Exception as e:
        print(f"Error setting up organization structure: {str(e)}")
        raise

def get_organization_summary(org_name: str) -> Dict:
    """Get a summary of an organization including all topics and sources"""
    try:
        org = get_organization_by_name(org_name)
        if not org:
            return {"error": f"Organization '{org_name}' not found"}
        
        org_id = org["uuid"]
        topics = get_topics_by_organization(org_id)
        
        summary = {
            "organization": org,
            "topics": [],
            "total_sources": 0
        }
        
        for topic in topics:
            topic_id = topic["uuid"]
            sources = get_sources_by_topic(topic_id)
            
            topic_summary = {
                "topic": topic,
                "sources": sources,
                "source_count": len(sources)
            }
            
            summary["topics"].append(topic_summary)
            summary["total_sources"] += len(sources)
        
        return summary
        
    except Exception as e:
        print(f"Error getting organization summary: {str(e)}")
        return {"error": str(e)}