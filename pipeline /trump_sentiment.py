"""
Trump social media sentiment analysis using Twitter-RoBERTa.
Processes Trump's Truth Social posts and computes sentiment scores.

Model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Trained specifically on tweets/social media
- More appropriate than FinBERT for Trump's informal style
- Output: negative, neutral, positive
"""

from pathlib import Path
from typing import List, Dict
import yaml

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline


class TrumpSentiment:
    """Twitter-RoBERTa sentiment analyzer for Trump posts."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize Twitter-RoBERTa model.
        
        Args:
            model_name: HuggingFace model name
        """
        
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Create pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            return_all_scores=True
        )
    
    
    def analyze_text(self, text: str, max_length: int = 512) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text (Trump post)
            max_length: Maximum token length
        
        Returns:
            Dict with positive, neutral, negative probabilities and net sentiment
        """
        if not text or len(text.strip()) == 0:
            return {
                'positive': 0.33,
                'neutral': 0.34,
                'negative': 0.33,
                'net_sentiment': 0.0
            }
        
        try:
            # Truncate using tokenizer to ensure it fits
            text = text[:max_length * 4]  # Pre-truncate characters
            
            # Run sentiment analysis with explicit truncation
            result = self.sentiment_pipeline(text, truncation=True, max_length=512)[0]
            
            # Extract probabilities (Twitter-RoBERTa uses: negative, neutral, positive)
            scores = {item['label'].lower(): item['score'] for item in result}
            
            # Calculate net sentiment
            net_sentiment = scores.get('positive', 0) - scores.get('negative', 0)
            
            return {
                'positive': scores.get('positive', 0),
                'neutral': scores.get('neutral', 0),
                'negative': scores.get('negative', 0),
                'net_sentiment': net_sentiment
            }
        
        except Exception as e:
            print(f"WARNING: Error analyzing text: {e}")
            return {
                'positive': 0.33,
                'neutral': 0.34,
                'negative': 0.33,
                'net_sentiment': 0.0
            }
    
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of Trump post texts
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with sentiment scores
        """
        
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                scores = self.analyze_text(text)
                results.append(scores)
        
        return pd.DataFrame(results)


def load_config(config_path: str = "config_hf.yaml") -> dict:
    """Load configuration."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}


def process_trump_posts(
    input_file: str = "data_files/trump_sentiment.csv",
    output_file: str = "data_files/trump_sentiment.csv",
    batch_size: int = 16
) -> pd.DataFrame:
    """
    Process Trump posts and compute sentiment scores.
    
    Args:
        input_file: Path to raw Trump posts CSV
        output_file: Path to save sentiment results
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with posts and sentiment scores
    """
    
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
    
    # Initialize sentiment analyzer
    sentiment_analyzer = TrumpSentiment()
    
    # Analyze sentiment
    sentiment_df = sentiment_analyzer.analyze_batch(
        texts=df['text'].tolist(),
        batch_size=batch_size
    )
    
    # Combine with original data
    result = pd.concat([df, sentiment_df], axis=1)
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    result.to_csv(output_path, index=False)
    
    return result


def main():
    import argparse
    from glob import glob
    import os
    
    # Find latest trump_raw file by modification time (not alphabetically)
    trump_files = glob('data_files/trump_raw_*.csv')
    if trump_files:
        default_input = max(trump_files, key=lambda x: (os.path.getsize(x), os.path.getmtime(x)))
    else:
        default_input = 'data_files/trump_raw_latest.csv'
    
    parser = argparse.ArgumentParser(
        description="Analyze sentiment of Trump's Truth Social posts"
    )
    parser.add_argument(
        '--input',
        type=str,
        default=default_input,
        help="Input CSV file with raw Trump posts (default: latest trump_raw_*.csv)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data_files/trump_sentiment.csv',
        help="Output CSV file for sentiment results"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    
    process_trump_posts(
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
