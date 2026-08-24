import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# 1. AUTO BUILD DATASET - no csv load
def build_dataset(base_url, max_pages=2):
    records = []
    for page in range(1, max_pages+1):
        url = f"{base_url}/page/{page}/"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.find_all('div', class_='quote'):
            text = item.find('span', class_='text').get_text(strip=True).replace('“', '').replace('”', '')
            author = item.find('small', class_='author').get_text(strip=True)
            tags = [tag.get_text(strip=True) for tag in item.find_all('a', class_='tag')]
            
            records.append({
                'text': text,
                'author': author,
                'tags': ', '.join(tags) if tags else 'Unknown'
            })
    
    df = pd.DataFrame(records)
    print(f"Auto-built dataset with {len(df)} records from {base_url}")
    return df

# Build it
df = build_dataset('http://quotes.toscrape.com', max_pages=2)

# 2. Inspect structure and data types
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

# 3. Clean duplicates and missing records
df.drop_duplicates(subset=['text'], inplace=True)
df['tags'] = df['tags'].fillna('Unknown')
df.dropna(subset=['text', 'author'], inplace=True)

# 4. Feature engineering for text metrics
df['char_count'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

print("\n--- Summary Statistics ---")
print(df[['char_count', 'word_count']].describe())

# 5. Anomaly / Outlier Detection using IQR
Q1 = df['word_count'].quantile(0.25)
Q3 = df['word_count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['word_count'] < lower_bound) | (df['word_count'] > upper_bound)]
print(f"\nDetected {len(outliers)} word-count outliers based on 1.5*IQR threshold.")

if not outliers.empty:
    print(outliers[['author', 'word_count', 'text']].head())

# 6. Save cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
print(f"\nCleaned dataset saved: {len(df)} records -> cleaned_data.csv")
print(df.head())
