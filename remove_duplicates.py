import csv
import shutil
from pathlib import Path

def remove_duplicates_from_csv_files(dedupe_strategy='title', backup=True):
    """
    Remove duplicate articles from CSV files in the data folder.
    
    Args:
        dedupe_strategy: 'title' - keep first occurrence of each title
                        'link' - keep first occurrence of each link
                        'both' - keep first occurrence of each title+link combination
                        'exact' - only remove exact duplicates (same title AND link)
        backup: If True, create backups before modifying files
    """
    # Define paths
    data_folder = Path("data")
    backup_folder = Path("data_backup")
    
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
    print(f"Deduplication strategy: {dedupe_strategy}")
    print("=" * 80)
    
    # Create backup folder if needed
    if backup:
        if not backup_folder.exists():
            backup_folder.mkdir()
            print(f"Created backup folder: {backup_folder}")
        print(f"Backups will be saved to: {backup_folder}")
        print()
    
    total_articles_before = 0
    total_articles_after = 0
    total_duplicates_removed = 0
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            # Backup original file
            if backup:
                backup_path = backup_folder / csv_file.name
                shutil.copy2(csv_file, backup_path)
            
            # Read all articles
            articles = []
            seen = set()
            duplicates_in_file = 0
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                
                for row in reader:
                    title = row.get('title', '').strip()
                    link = row.get('link', '').strip()
                    
                    # Determine uniqueness key based on strategy
                    if dedupe_strategy == 'title':
                        unique_key = title
                    elif dedupe_strategy == 'link':
                        unique_key = link
                    elif dedupe_strategy == 'both':
                        unique_key = f"{title}|||{link}"
                    elif dedupe_strategy == 'exact':
                        unique_key = f"{title}|||{link}"
                    else:
                        raise ValueError(f"Invalid strategy: {dedupe_strategy}")
                    
                    # Check if we've seen this before
                    if unique_key not in seen:
                        seen.add(unique_key)
                        articles.append(row)
                    else:
                        duplicates_in_file += 1
            
            articles_before = len(articles) + duplicates_in_file
            articles_after = len(articles)
            
            # Write cleaned data back to file
            with open(csv_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(articles)
            
            print(f"  Before: {articles_before} articles")
            print(f"  After: {articles_after} articles")
            print(f"  Removed: {duplicates_in_file} duplicates")
            print()
            
            total_articles_before += articles_before
            total_articles_after += articles_after
            total_duplicates_removed += duplicates_in_file
        
        except Exception as e:
            print(f"  ❌ Error processing {csv_file.name}: {e}")
            print()
            continue
    
    print("=" * 80)
    print("SUMMARY:")
    print(f"  Total articles before: {total_articles_before:,}")
    print(f"  Total articles after: {total_articles_after:,}")
    print(f"  Total duplicates removed: {total_duplicates_removed:,}")
    print(f"  Reduction: {(total_duplicates_removed/total_articles_before*100):.2f}%")
    
    if backup:
        print(f"\n  ✅ Original files backed up to: {backup_folder}/")
    
    print(f"\n  ✅ All files have been cleaned!")

def main():
    """
    Main function with user options.
    """
    print("CSV Duplicate Remover")
    print("=" * 80)
    print("\nDeduplication strategies:")
    print("  1. 'title'  - Remove articles with duplicate titles (keep first occurrence)")
    print("  2. 'link'   - Remove articles with duplicate links (keep first occurrence)")
    print("  3. 'both'   - Remove articles with duplicate title OR link (most aggressive)")
    print("  4. 'exact'  - Only remove exact duplicates (same title AND link)")
    print()
    print("Default: Using 'title' strategy (most common use case)")
    print()
    
    # Use 'title' strategy by default
    strategy = 'title'
    
    remove_duplicates_from_csv_files(dedupe_strategy=strategy, backup=True)

if __name__ == "__main__":
    main()



