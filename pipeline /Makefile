# Data Pipeline Makefile
.PHONY: all clean rebuild help

all:
	@echo "Running complete pipeline..."
	python fetch_ohlcv.py
	python news_data.py --config config_hf.yaml
	python news_sentiment.py --config config_hf.yaml
	python news_features.py --config config_hf.yaml
	python garch_features.py --config config_hf.yaml
	python trump_data.py
	python trump_sentiment.py
	python trump_features.py
	python create_labels.py --config config_hf.yaml
	python combine_features.py
	@echo "Pipeline complete"

clean:
	rm -f data_files/*.csv

rebuild: clean all

help:
	@echo "Usage:"
	@echo "  make        - Run complete pipeline"
	@echo "  make clean  - Remove all CSV files"
	@echo "  make rebuild - Clean and rebuild everything"
