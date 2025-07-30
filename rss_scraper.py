import feedparser
import pandas as pd
from datetime import datetime
import os
from newspaper import Article

# === CONFIG ===
scrape_content = False  # Set to True if you want full article scraping
today = datetime.today().strftime("%Y-%m-%d")
csv_path = f"data/articles_{today}.csv"

# === RSS Feed List ===
rss_feeds = {
    "MarketWatch Top": "https://www.marketwatch.com/rss/topstories",
    "CNBC Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Investing.com All News": "https://uk.investing.com/rss/news.rss",
    "Investing.com Bond News": "https://uk.investing.com/rss/bonds.rss",
    "Google News Reuters" : "https://news.google.com/rss/search?q=site%3Areuters.com&hl=en-US&gl=US&ceid=US%3Aen",
    "Google News WSJ" : "https://news.google.com/rss/search?q=site:wsj.com&hl=en-US&gl=US&ceid=US:en",
    "Google News FT" : "https://news.google.com/rss/search?q=site:ft.com&hl=en-US&gl=US&ceid=US:en",
}

# === Scraping Function ===
def scrape_article_content(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

# === Main Scraper ===
def main():
    os.makedirs("data", exist_ok=True)

    # Load existing articles (if today's CSV exists)
    if os.path.exists(csv_path):
        seen_df = pd.read_csv(csv_path)
        seen_links = set(seen_df['link'])
    else:
        seen_df = pd.DataFrame()
        seen_links = set()

    new_articles = []

    for source, url in rss_feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if entry.link not in seen_links:
                content = scrape_article_content(entry.link) if scrape_content else ""
                new_articles.append({
                    "timestamp": datetime.now().isoformat(timespec='seconds'),
                    "source": source,
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", "N/A"),
                    "content": content
                })
                seen_links.add(entry.link)

    if new_articles:
        df_new = pd.DataFrame(new_articles)
        df_combined = pd.concat([seen_df, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        print(f"✅ {len(new_articles)} new articles saved.")
    else:
        print("No new articles found.")

# === Entry Point ===
if __name__ == "__main__":
    main()
