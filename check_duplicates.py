import csv
from pathlib import Path
from collections import defaultdict

def check_duplicates_in_csv_files():
    """
    Check each CSV file in the data folder for duplicate articles.
    Reports duplicates based on title, link, and exact matches.
    """
    # Define paths
    data_folder = Path("data")
    output_file = "duplicate_report.txt"
    
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
    print("=" * 80)
    
    # Store results for the report
    report_lines = []
    total_files_with_duplicates = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        try:
            # Track duplicates
            title_counts = defaultdict(list)
            link_counts = defaultdict(list)
            exact_match_counts = defaultdict(list)
            
            articles = []
            
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for idx, row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                    articles.append((idx, row))
                    
                    title = row.get('title', '').strip()
                    link = row.get('link', '').strip()
                    
                    if title:
                        title_counts[title].append(idx)
                    if link:
                        link_counts[link].append(idx)
                    
                    # Create a key for exact matches (title + link)
                    exact_key = f"{title}|||{link}"
                    exact_match_counts[exact_key].append(idx)
            
            # Find duplicates
            duplicate_titles = {k: v for k, v in title_counts.items() if len(v) > 1}
            duplicate_links = {k: v for k, v in link_counts.items() if len(v) > 1}
            duplicate_exact = {k: v for k, v in exact_match_counts.items() if len(v) > 1}
            
            total_articles = len(articles)
            has_duplicates = False
            
            # Generate report for this file
            file_report = []
            file_report.append(f"\n{'=' * 80}")
            file_report.append(f"FILE: {csv_file.name}")
            file_report.append(f"Total articles: {total_articles}")
            file_report.append(f"{'=' * 80}")
            
            if duplicate_exact:
                has_duplicates = True
                file_report.append(f"\n🔴 EXACT DUPLICATES (same title AND link): {len(duplicate_exact)}")
                for exact_key, row_nums in sorted(duplicate_exact.items(), key=lambda x: len(x[1]), reverse=True):
                    title, link = exact_key.split('|||')
                    file_report.append(f"\n  Found {len(row_nums)} times (rows: {row_nums}):")
                    file_report.append(f"    Title: {title[:100]}{'...' if len(title) > 100 else ''}")
                    file_report.append(f"    Link: {link[:100]}{'...' if len(link) > 100 else ''}")
            
            if duplicate_titles:
                has_duplicates = True
                # Subtract exact duplicates to avoid double counting
                title_only_dups = {k: v for k, v in duplicate_titles.items() 
                                  if not any(exact_key.startswith(k + '|||') for exact_key in duplicate_exact)}
                
                if title_only_dups:
                    file_report.append(f"\n🟡 DUPLICATE TITLES (different links): {len(title_only_dups)}")
                    for title, row_nums in sorted(title_only_dups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                        file_report.append(f"\n  Found {len(row_nums)} times (rows: {row_nums}):")
                        file_report.append(f"    Title: {title[:100]}{'...' if len(title) > 100 else ''}")
            
            if duplicate_links:
                has_duplicates = True
                # Subtract exact duplicates to avoid double counting
                link_only_dups = {k: v for k, v in duplicate_links.items() 
                                 if not any(exact_key.endswith('|||' + k) for exact_key in duplicate_exact)}
                
                if link_only_dups:
                    file_report.append(f"\n🟡 DUPLICATE LINKS (different titles): {len(link_only_dups)}")
                    for link, row_nums in sorted(link_only_dups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
                        file_report.append(f"\n  Found {len(row_nums)} times (rows: {row_nums}):")
                        file_report.append(f"    Link: {link[:100]}{'...' if len(link) > 100 else ''}")
            
            if not has_duplicates:
                file_report.append(f"\n✅ No duplicates found!")
                print(f"  ✅ No duplicates")
            else:
                total_files_with_duplicates += 1
                print(f"  ⚠️  Found duplicates - Exact: {len(duplicate_exact)}, Title: {len(duplicate_titles)}, Link: {len(duplicate_links)}")
                report_lines.extend(file_report)
        
        except Exception as e:
            error_msg = f"\n❌ Error processing {csv_file.name}: {e}"
            print(error_msg)
            report_lines.append(error_msg)
    
    # Write detailed report to file
    print(f"\n{'=' * 80}")
    print(f"\nWriting detailed report to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("DUPLICATE ARTICLES REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total files checked: {len(csv_files)}\n")
        f.write(f"Files with duplicates: {total_files_with_duplicates}\n")
        f.write(f"Files without duplicates: {len(csv_files) - total_files_with_duplicates}\n")
        f.write("\n")
        
        if report_lines:
            f.write("\n".join(report_lines))
        else:
            f.write("\n✅ No duplicates found in any files!\n")
    
    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"  Total files checked: {len(csv_files)}")
    print(f"  Files with duplicates: {total_files_with_duplicates}")
    print(f"  Files without duplicates: {len(csv_files) - total_files_with_duplicates}")
    print(f"\nDetailed report saved to: {output_file}")

if __name__ == "__main__":
    check_duplicates_in_csv_files()



