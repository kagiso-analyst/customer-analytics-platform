@"
# Data Directory

This directory contains generated data files that are ignored by Git.

## Files generated when running the project:
- \`raw/customers.csv\` - Sample customer data
- \`raw/transactions.csv\` - Sample transaction data  
- \`customer_analytics.db\` - SQLite database with processed data

## To regenerate data:
\`\`\`bash
python 1_generate_data.py
python src/data/etl_pipeline.py
\`\`\`
"@ | Out-File -FilePath "data/README.md" -Encoding UTF8

# Create models/README.md
@"
# Models Directory

This directory contains trained machine learning models that are ignored by Git.

## Models generated:
- \`clv_model.pkl\` - Customer Lifetime Value prediction model

## To regenerate models:
Run the dashboard or analytics scripts to train new models.
"@ | Out-File -FilePath "models/README.md" -Encoding UTF8