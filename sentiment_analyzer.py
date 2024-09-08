import requests
from textblob import TextBlob
from newspaper import Article
import nltk

nltk.download('punkt')

def fetch_news(ticker, api_key, num_articles=5):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}&language=en&sortBy=publishedAt&pageSize={num_articles}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_company_sentiment(ticker, api_key):
    articles = fetch_news(ticker, api_key)
    sentiments = []

    for article in articles:
        try:
            article_obj = Article(article['url'])
            article_obj.download()
            article_obj.parse()
            article_obj.nlp()
            sentiment = analyze_sentiment(article_obj.summary)
            sentiments.append(sentiment)
        except Exception as e:
            print(f"Error processing article: {e}")

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        return avg_sentiment
    else:
        return 0

def interpret_sentiment(sentiment_score):
    if sentiment_score > 0.05:
        return "Positive"
    elif sentiment_score < -0.05:
        return "Negative"
    else:
        return "Neutral"