#!/bin/bash
set -e

echo "================================"
echo "RUNNING COMPLETE DATA PIPELINE"
echo "================================"

export FINNHUB_API_KEY=d46r251r01qgc9eu5phgd46r251r01qgc9eu5pi0

echo ""
echo "Step 1: Fetching OHLCV data..."
python fetch_ohlcv.py

echo ""
echo "Step 2: Fetching news data..."
python news_data.py --config config_hf.yaml

echo ""
echo "Step 3: Processing sentiment..."
python news_sentiment.py --config config_hf.yaml

echo ""
echo "Step 4: Creating news features..."
python news_features.py --config config_hf.yaml

echo ""
echo "Step 5: Creating GARCH features..."
python garch_features.py --config config_hf.yaml

echo ""
echo "Step 6: Creating regime labels..."
python create_labels.py --config config_hf.yaml

echo ""
echo "Step 7: Combining all features..."
python combine_features.py

echo ""
echo "================================"
echo "PIPELINE COMPLETE!"
echo "================================"
ls -lh data_files/*.csv
