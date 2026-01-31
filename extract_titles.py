import csv
import os
import re
from pathlib import Path

def clean_title(title):
    """
    Remove news source names from the end of headlines.
    Handles patterns like " - Reuters", " - The Wall Street Journal", etc.
    """
    # Common patterns: " - Source", " | Source", "- Source" at the end
    # Use regex to match and remove source names at the end
    patterns = [
        r'\s*[-–—|]\s*Reuters\s*$',
        r'\s*[-–—|]\s*Financial Times\s*$',
        r'\s*[-–—|]\s*The Wall Street Journal\s*$',
        r'\s*[-–—|]\s*WSJ\s*$',
        r'\s*[-–—|]\s*Bloomberg\s*$',
        r'\s*[-–—|]\s*MarketWatch\s*$',
        r'\s*[-–—|]\s*CNBC\s*$',
        r'\s*[-–—|]\s*CNN\s*$',
        r'\s*[-–—|]\s*BBC\s*$',
        r'\s*[-–—|]\s*AP News\s*$',
        r'\s*[-–—|]\s*Associated Press\s*$',
        r'\s*[-–—|]\s*Forbes\s*$',
        r'\s*[-–—|]\s*Barron\'s\s*$',
        r'\s*[-–—|]\s*Investor\'s Business Daily\s*$',
        r'\s*[-–—|]\s*The Motley Fool\s*$',
        r'\s*[-–—|]\s*Seeking Alpha\s*$',
        r'\s*[-–—|]\s*Yahoo Finance\s*$',
        r'\s*[-–—|]\s*New York Times\s*$',
        r'\s*[-–—|]\s*The New York Times\s*$',
        r'\s*[-–—|]\s*Washington Post\s*$',
        r'\s*[-–—|]\s*The Washington Post\s*$',
        r'\s*[-–—|]\s*Guardian\s*$',
        r'\s*[-–—|]\s*The Guardian\s*$',
        r'\s*[-–—|]\s*FT\s*$',
    ]
    
    cleaned = title
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Clean up any trailing whitespace or punctuation
    cleaned = cleaned.strip()
    
    return cleaned

def extract_titles_from_csv_files():
    """
    Extract titles from all CSV files in the data folder and save to a text file.
    Removes news source names from the end of headlines.
    """
    # Define paths
    data_folder = Path("data")
    output_file = "extracted_titles.txt"
    
    # Check if data folder exists
    if not data_folder.exists():
        print(f"Error: {data_folder} folder not found!")
        return
    
    # Get all CSV files in the data folder
    csv_files = sorted(data_folder.glob("articles_*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_folder}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Extract titles
    all_titles = []
    sources_removed = 0
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    if 'title' in row and row['title']:
                        original_title = row['title']
                        cleaned_title = clean_title(original_title)
                        all_titles.append(cleaned_title)
                        
                        if original_title != cleaned_title:
                            sources_removed += 1
        
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Write titles to text file
    print(f"\nWriting {len(all_titles)} titles to {output_file}...")
    print(f"Removed source names from {sources_removed} headlines")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for title in all_titles:
            f.write(title + '\n')
    
    print(f"Done! Extracted {len(all_titles)} titles to {output_file}")

if __name__ == "__main__":
    extract_titles_from_csv_files()

