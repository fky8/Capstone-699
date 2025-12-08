"""
News sentiment analysis using FinBERT.
Processes news articles and computes sentiment scores.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline



class FinBERTSentiment:
    """FinBERT sentiment analyzer."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT model.
        
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
            text: Input text
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
        
        # Truncate text if too long
        text = text[:max_length * 4]  # Rough char estimate
        
        try:
            # Run sentiment analysis
            result = self.sentiment_pipeline(text)[0]
            
            # Extract probabilities
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
            texts: List of texts
            batch_size: Batch size for processing
        
        Returns:
            DataFrame with sentiment scores
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                sentiment = self.analyze_text(text)
                results.append(sentiment)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i // batch_size + 1} batches...")
        
        return pd.DataFrame(results)


def process_news_sentiment(
    df: pd.DataFrame,
    text_columns: List[str] = ['title', 'summary']
) -> pd.DataFrame:
    """
    Process news articles and add sentiment scores.
    
    Args:
        df: News DataFrame
        text_columns: Columns to combine for sentiment analysis
    
    Returns:
        DataFrame with sentiment columns added
    """
    
    # Initialize FinBERT
    sentiment_analyzer = FinBERTSentiment()
    
    # Combine text columns
    df['text_combined'] = df[text_columns[0]].fillna('')
    for col in text_columns[1:]:
        df['text_combined'] += ' ' + df[col].fillna('')
    
    # Analyze sentiment
    sentiment_df = sentiment_analyzer.analyze_batch(df['text_combined'].tolist())
    
    # Add sentiment columns
    df['sent_positive'] = sentiment_df['positive']
    df['sent_neutral'] = sentiment_df['neutral']
    df['sent_negative'] = sentiment_df['negative']
    df['sent_net'] = sentiment_df['net_sentiment']
    
    # Drop temporary column
    df = df.drop(columns=['text_combined'])
    
    
    return df


def save_sentiment_data(df: pd.DataFrame, output_path: str) -> None:
    """Save sentiment data to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_sentiment_data(input_path: str) -> pd.DataFrame:
    """Load sentiment data from CSV."""
    df = pd.read_csv(input_path, parse_dates=['datetime'])
    # Ensure datetime is UTC
    if df['datetime'].dt.tz is None:
        df['datetime'] = df['datetime'].dt.tz_localize('UTC')
    return df


def main():
    """CLI entry point for sentiment analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze news sentiment using FinBERT")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input news CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output sentiment CSV file"
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get input/output paths
    data_dir = config['output']['data_dir']
    input_path = args.input or f"{data_dir}/master_news_raw.csv"
    output_path = args.output or f"{data_dir}/news_sentiment.csv"
    
    # Load news data
    from news_data import load_news_data
    df = load_news_data(input_path)
    
    if df.empty:
        return
    
    # Process sentiment
    df = process_news_sentiment(df)
    
    # Save results
    save_sentiment_data(df, output_path)
    
    # Display summary
    
    # Show most positive and negative articles
    top_positive = df.nlargest(3, 'sent_net')[['datetime', 'ticker', 'title', 'sent_net']]
    
    top_negative = df.nsmallest(3, 'sent_net')[['datetime', 'ticker', 'title', 'sent_net']]


if __name__ == "__main__":
    main()
