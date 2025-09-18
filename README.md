# Better Rules - PDF Processing Pipeline

Extract pin tables, ordering info, and design rules from microcontroller datasheets.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create `.env` file:**
   ```bash
   # AI Processing (at least one required)
   GOOGLE_API_KEY=your_google_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here

   # Database (required)
   MONGODB_URI=mongodb://localhost:27017/your_database_name
   ```

## Usage

1. **Place PDF datasheets** in the `datasheets/` folder
2. **Run extraction:**
   ```bash
   python run.py
   ```
3. **View results** in the `outputs/` folder

## Output

Each processed PDF generates:
- Individual JSON file with pin table, ordering info, and design rules
- `processing_summary.json` with overall statistics