import boto3
import os
from typing import List, Dict, Any
from botocore.exceptions import ClientError
import json

class S3DatasheetManager:
    def __init__(self, base_prefix: str):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'didy-bucket'
        self.base_prefix = base_prefix
        self.output_prefix = f'{self.base_prefix}output/'
    
    def list_child_folders(self) -> List[str]:
        """List the 3 existing child folders in the S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=self.base_prefix,
                Delimiter='/'
            )
            
            folders = []
            for prefix in response.get('CommonPrefixes', []):
                folder_name = prefix['Prefix'].replace(self.base_prefix, '').rstrip('/')
                if folder_name and folder_name != 'output':
                    folders.append(folder_name)
            
            return folders
        except ClientError as e:
            print(f"Error listing folders: {e}")
            return []
    
    def list_datasheets_in_folder(self, folder_name: str) -> List[str]:
        """List all datasheet files in a specific child folder."""
        try:
            prefix = f'{self.base_prefix}{folder_name}/'
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            datasheets = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith(('.pdf', '.PDF')):
                    datasheets.append(key)
            
            return datasheets
        except ClientError as e:
            print(f"Error listing datasheets in {folder_name}: {e}")
            return []
    
    def download_datasheet(self, s3_key: str, local_path: str) -> bool:
        """Download a datasheet from S3 to local file."""
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            return True
        except ClientError as e:
            print(f"Error downloading {s3_key}: {e}")
            return False
    
    def read_datasheet_content(self, s3_key: str) -> bytes:
        """Read datasheet content directly from S3."""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return response['Body'].read()
        except ClientError as e:
            print(f"Error reading {s3_key}: {e}")
            return b''
    
    def create_output_folder(self) -> bool:
        """Create the output folder in S3 by uploading a placeholder file."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=f'{self.output_prefix}.gitkeep',
                Body=b''
            )
            return True
        except ClientError as e:
            print(f"Error creating output folder: {e}")
            return False
    
    def processed_file_exists(self, datasheet_key: str) -> bool:
        """Check if processed JSON file already exists for this datasheet."""
        try:
            datasheet_name = os.path.basename(datasheet_key).replace('.pdf', '').replace('.PDF', '')
            output_key = f'{self.output_prefix}{datasheet_name}_processed.json'
            
            self.s3_client.head_object(Bucket=self.bucket_name, Key=output_key)
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                print(f"Error checking if processed file exists for {datasheet_key}: {e}")
                return False
    
    def write_processed_datasheet(self, datasheet_name: str, processed_data: Dict[str, Any]) -> bool:
        """Write processed datasheet data (pin table, ordering info, rules) to output folder."""
        try:
            output_key = f'{self.output_prefix}{datasheet_name}_processed.json'
            
            json_data = json.dumps(processed_data, indent=2, ensure_ascii=False)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=output_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            
            print(f"Successfully wrote processed data to {output_key}")
            return True
            
        except ClientError as e:
            print(f"Error writing processed datasheet {datasheet_name}: {e}")
            return False
    
    def get_all_datasheets(self) -> Dict[str, List[str]]:
        """Get all datasheets organized by folder."""
        all_datasheets = {}
        folders = self.list_child_folders()
        
        for folder in folders:
            datasheets = self.list_datasheets_in_folder(folder)
            all_datasheets[folder] = datasheets
            
        return all_datasheets
    
    def process_and_upload_datasheet(self, s3_key: str, pin_table: List[Dict], ordering_info: Dict, extracted_rules: List[str]) -> bool:
        """Process a datasheet and upload the results to the output folder."""
        datasheet_name = os.path.basename(s3_key).replace('.pdf', '').replace('.PDF', '')
        
        processed_data = {
            'source_file': s3_key,
            'pin_table': pin_table,
            'ordering_info': ordering_info,
            'extracted_rules': extracted_rules,
            'processed_at': str(os.popen('date').read().strip())
        }
        
        return self.write_processed_datasheet(datasheet_name, processed_data)

if __name__ == "__main__":
    manager = S3DatasheetManager('St.com/Microcontrollers---microprocessors/')
    
    print("Creating output folder...")
    manager.create_output_folder()
    
    print("Getting all datasheets...")
    all_datasheets = manager.get_all_datasheets()
    
    for folder, datasheets in all_datasheets.items():
        print(f"\nFolder: {folder}")
        print(f"Datasheets: {len(datasheets)}")
        for datasheet in datasheets[:3]:  # Show first 3
            print(f"  - {datasheet}")