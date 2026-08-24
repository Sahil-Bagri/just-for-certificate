import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- 1. AUTO BUILD DATASET ---
def build_dataset(base_url, max_pages=2):
    records = []
    for page in range(1, max_pages+1):
        soup = BeautifulSoup(requests.get(f"{base_url}/page/{page}/").text, 'html.parser')
        for item in soup.find_all('div', class_='quote'):
            records.append({
                'text': item.find('span', class_='text').get_text(strip=True).replace('“', '').replace('”', ''),
                'author': item.find('small', class_='author').get_text(strip=True),
                'tags': ', '.join([t.get_text() for t in item.find_all('a', class_='tag')])
            })
    df = pd.DataFrame(records)
    df.drop_duplicates(subset=['text'], inplace=True)
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    return df

df = build_dataset('http://quotes.toscrape.com', max_pages=3)

# --- 2. VISUALIZATIONS ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Word Count Distribution
sns.histplot(df['word_count'], kde=True, color='teal', ax=axes[0, 0])
axes[0, 0].set_title('Word Count Distribution', fontsize=14, weight='bold')

# 2. Top 10 Authors
top_authors = df['author'].value_counts().head(10)
sns.barplot(x=top_authors.values, y=top_authors.index, palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('Top 10 Authors by Frequency', fontsize=14, weight='bold')

# 3. Boxplot Top 5 Authors
top_5_authors = df['author'].value_counts().head(5).index
filtered_df = df[df['author'].isin(top_5_authors)]
sns.boxplot(data=filtered_df, x='author', y='char_count', palette='Set2', ax=axes[1, 0])
axes[1, 0].set_title('Character Count Spread (Top 5 Authors)', fontsize=14, weight='bold')
axes[1, 0].tick_params(axis='x', rotation=20)

# 4. WordCloud
all_text = ' '.join(df['text'].dropna())
wordcloud = WordCloud(width=600, height=400, background_color='white', colormap='plasma').generate(all_text)
axes[1, 1].imshow(wordcloud, interpolation='bilinear')
axes[1, 1].axis('off')
axes[1, 1].set_title('Word Cloud of Text Data', fontsize=14, weight='bold')

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300)
plt.show()
print("Saved as eda_visualizations.png")
