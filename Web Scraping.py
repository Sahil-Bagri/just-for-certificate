import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_data(base_url, max_pages=3):
    records = []
    
    for page in range(1, max_pages + 1):
        url = f"{base_url}/page/{page}/"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            break
            
        soup = BeautifulSoup(response.text, 'html.parser')
        quotes = soup.find_all('div', class_='quote')
        
        for item in quotes:
            text = item.find('span', class_='text').get_text(strip=True).replace('“', '').replace('”', '')
            author = item.find('small', class_='author').get_text(strip=True)
            tags = [tag.get_text(strip=True) for tag in item.find_all('a', class_='tag')]
            
            records.append({
                'text': text,
                'author': author,
                'tags': ', '.join(tags) if tags else 'general'
            })
            
    df = pd.DataFrame(records)
    df.to_csv('scraped_data.csv', index=False)
    print(f"Successfully scraped {len(df)} records to 'scraped_data.csv'.")
    return df

if __name__ == '__main__':
    BASE_URL = 'http://quotes.toscrape.com'
    df = scrape_data(BASE_URL, max_pages=5)
    print(df.head())
