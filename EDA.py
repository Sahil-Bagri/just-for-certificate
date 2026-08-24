import requests
from bs4 import BeautifulSoup
import pandas as pd

# Auto build from website directly 
def build_dataset(base_url, max_pages=2):
    records = []
    for page in range(1, max_pages+1):
        soup = BeautifulSoup(requests.get(f"{base_url}/page/{page}/").text, 'html.parser')
        for item in soup.find_all('div', class_='quote'):
            records.append({
                'text': item.find('span', class_='text').get_text(strip=True),
                'author': item.find('small', class_='author').get_text(strip=True),
                'tags': ', '.join([t.get_text() for t in item.find_all('a', class_='tag')])
            })
    return pd.DataFrame(records)

df = build_dataset('http://quotes.toscrape.com', max_pages=2)
import pandas as pd
import numpy as np

# 1. Load dataset
df = pd.read_csv('scraped_data.csv')

# 2. Inspect structure and data types
print("--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 3. Clean duplicates and missing records
df.drop_duplicates(subset=['text'], inplace=True)
df['tags'] = df['tags'].fillna('Unknown')

# 4. Feature engineering for text metrics
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\n--- Summary Statistics ---")
print(df[['char_count', 'word_count']].describe())

# 5. Anomaly / Outlier Detection using Interquartile Range (IQR)
Q1 = df['word_count'].quantile(0.25)
Q3 = df['word_count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]
print(f"\nDetected {len(outliers)} word-count outliers based on 1.5*IQR threshold.")

# Save cleaned dataset for downstream analysis
df.to_csv('cleaned_data.csv', index=False)
