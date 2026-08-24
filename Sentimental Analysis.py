import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('vader_lexicon', quiet=True)

# 1. Text Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = pd.read_csv('cleaned_data.csv')
df['cleaned_text'] = df['text'].apply(clean_text)

# 2. VADER Sentiment Scoring
sia = SentimentIntensityAnalyzer()

df['compound'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
df['pos_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['pos'])
df['neu_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['neu'])
df['neg_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['neg'])

# 3. Label Classification
def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['compound'].apply(classify_sentiment)

print("--- Sentiment Distribution ---")
print(df['sentiment'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')

# 4. Save Final Dataset with Sentiment Labels
df.to_csv('final_sentiment_analysis.csv', index=False)

# 5. Visualizing Sentiment Breakdown
plt.figure(figsize=(8, 5))
palette_colors = {'Positive': '#2ecc71', 'Neutral': '#95a5a6', 'Negative': '#e74c3c'}
sns.countplot(data=df, x='sentiment', order=['Positive', 'Neutral', 'Negative'], palette=palette_colors)
plt.title('Sentiment Distribution Across Dataset', fontsize=14, weight='bold')
plt.xlabel('Sentiment Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('sentiment_distribution.png', dpi=300)
plt.show()
