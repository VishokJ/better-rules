# Better Rules - High-Performance PDF Processing Pipeline

Super powerful, super small async PDF processing system for extracting pin tables, ordering info, and design rules from STM microcontroller datasheets.

## 🚀 One-Command EC2 Setup

```bash
curl -sSL https://raw.githubusercontent.com/yourusername/better-rules/main/setup_ec2.sh | bash
```

## 🏃‍♂️ Quick Start

### Local Development
```bash
git clone <repo-url>
cd better-rules
pip install -r requirements.txt
cp .env.template .env  # Update with your credentials
python main.py --workers 10 --test
```

### Production (EC2)
```bash
# After running setup_ec2.sh
sudo systemctl start better-rules
# or
./run.sh --workers 100
```

## 📊 Usage Examples

### Full Pipeline (100 workers)
```bash
python main.py --workers 100 --filter-prefix STM
```

### Pin Extraction Only
```bash
python main.py --workers 50 --stage pin-only --filter-prefix STM
```

### Rule Generation Only (for completed pin extractions)
```bash
python main.py --workers 50 --stage rules-only
```

### Test Mode (5 files only)
```bash
python main.py --workers 10 --test
```

## 🏗️ Architecture

**Two-Stage Async Pipeline:**
1. **Stage 1**: Extract pin tables & ordering info from STM PDFs
2. **Stage 2**: Generate design rules from PDFs that completed Stage 1

**Key Features:**
- 100 configurable async workers
- S3 integration for input/output
- Comprehensive error handling & logging
- Real-time progress tracking
- Automatic temp file cleanup

## 📁 File Structure

```
better-rules/
├── main.py              # Main async pipeline
├── documents.py         # S3DatasheetManager
├── extraction.py        # RuleExtractor
├── functions.py         # Pin table extraction
├── setup_ec2.sh        # One-command EC2 setup
├── run.sh              # Production runner
└── requirements.txt    # Dependencies
```

## ⚙️ Configuration

Environment variables in `.env`:
- `MAX_WORKERS`: Default worker count (100)
- `GOOGLE_API_KEY`: For Gemini API
- `MONGODB_URI`: MongoDB connection
- `AWS_ACCESS_KEY` / `AWS_SECRET_KEY`: S3 access
- `SUPABASE_*`: Database credentials

## 📈 Performance

**Benchmarks on EC2 t3.2xlarge:**
- Stage 1 (Pin extraction): ~15s per PDF
- Stage 2 (Rule generation): ~30s per PDF
- 100 workers: Process ~240 PDFs/hour

## 🔧 Monitoring

```bash
# System service logs
sudo journalctl -u better-rules -f

# Application logs
tail -f /var/log/better-rules/app.log

# Resource monitoring
htop
```

## 📦 S3 Structure

```
s3://didy-bucket/St.com/Microcontrollers---microprocessors/
├── folder1/          # Input PDFs
├── folder2/          # Input PDFs  
├── folder3/          # Input PDFs
└── output/           # Processed JSON files
    ├── STM32F401_processed.json
    └── STM32L476_processed.json
```

## 🎯 Output Format

Each processed datasheet produces:
```json
{
  "source_file": "s3://bucket/path/STM32F401.pdf",
  "pin_table": [...],
  "ordering_info": {...},
  "extracted_rules": [...],
  "processed_at": "2024-09-09T10:30:00"
}
```

## 🛠️ Development

### Adding New Extractors
1. Extend `RuleExtractor` class in `extraction.py`
2. Update `RELEVANT_SECTIONS` in `functions.py`
3. Test with `--test` flag

### Custom Filters
Modify `get_datasheets_by_prefix()` in `main.py` for different filtering logic.

## 📞 Support

- Logs: `/var/log/better-rules/app.log`
- Config: `.env`
- Service: `sudo systemctl status better-rules`