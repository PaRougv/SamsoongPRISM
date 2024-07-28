# Import necessary libraries
from datasets import load_dataset
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import spacy
"""
#To download the stopwords such as to, the , and etc 
#punkt is to tokenize and vectorize the lines of news articles and give the output , it is pretrained to works wonders
#wordnet is for the word cloud , also it provides antonyms synonys etc
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# to directly load the data set , you need to pip install datasets and then load the dataset as follows :-
ds = load_dataset("fancyzhx/ag_news")

#converting the dataset into DataFrame for easy analysis
train_df = pd.DataFrame(ds['train'])
test_df = pd.DataFrame(ds['test'])


#Converting to CSV and Loading the file
train_df.to_csv('ag_news_train.csv', index=False)
test_df.to_csv('ag_news_test.csv', index=False)

train_df_from_csv = pd.read_csv('ag_news_train.csv')
test_df_from_csv = pd.read_csv('ag_news_test.csv')

#Later in the code , we use SPACY - a library to do a lot of stuff for the same we have downloaded this over here. Ill explain this later
nlp = spacy.load('en_core_web_sm')

#this is the start of the univariate Analysis of the dataset


#in the next 2 lines we're taking the length of each and every news article and storing it in the column - 'article length'
train_df_from_csv['article_length'] = train_df_from_csv['text'].apply(len)
test_df_from_csv['article_length'] = test_df_from_csv['text'].apply(len)

train_article_length_skewness = skew(train_df_from_csv['article_length'])
train_article_length_kurtosis = kurtosis(train_df_from_csv['article_length'])
test_article_length_skewness = skew(test_df_from_csv['article_length'])
test_article_length_kurtosis = kurtosis(test_df_from_csv['article_length'])

print(f"Skewness of train article lengths: {train_article_length_skewness}")
print(f"Kurtosis of train article lengths: {train_article_length_kurtosis}")
print(f"Skewness of test article lengths: {test_article_length_skewness}")
print(f"Kurtosis of test article lengths: {test_article_length_kurtosis}")



#Finding the cosine similarity , something for the recommender system
#we took the first 100 headings , vectorized it , next - >This line applies the fit_transform method of the TfidfVectorizer to the text column of subset_df. This method learns the vocabulary and inverse document frequency (IDF) from the text data and transforms the text into a matrix of TF-IDF features. Each row in tfidf_matrix corresponds to an article, and each column corresponds to a term in the corpus. The values in the matrix represent the TF-IDF score of each term in each article.
#then we're finding the cosine similarity and then finally plotting everything at once
subset_df = train_df.head(100)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(subset_df['text'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=subset_df.index, columns=subset_df.index)

plt.figure(figsize=(10, 8))
sns.heatmap(cosine_sim_df, cmap="coolwarm", xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Heatmap')
plt.show()


#Analysis to find the average number of words in each news article and plotting the graph for it
#in the first line , we're taking the number of words in each artile and storing a count and then finally finding the mean
train_df['num_words'] = train_df['text'].apply(lambda x: len(x.split()))
average_words_per_article = train_df['num_words'].mean()
print(f"Average number of words per article: {average_words_per_article}")


#plotting can be gpted , its not that much
plt.figure(figsize=(12, 6))
sns.histplot(train_df['num_words'], bins=30, kde=True)
plt.title('Distribution of Number of Words per Article')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.axvline(average_words_per_article, color='r', linestyle='dashed', linewidth=2, label=f'Average: {average_words_per_article:.2f}')
plt.legend()
plt.show()



stop_words = set(stopwords.words('english'))

def count_and_list_stopwords(text):
    words = word_tokenize(text)
    stopword_list = [word.lower() for word in words if word.lower() in stop_words]
    return len(stopword_list), stopword_list

train_df['num_stopwords'], train_df['stopwords'] = zip(*train_df['text'].apply(count_and_list_stopwords))

plt.figure(figsize=(12, 6))
sns.histplot(train_df['num_stopwords'], bins=30, kde=True)
plt.title('Distribution of Number of Stopwords per Article')
plt.xlabel('Number of Stopwords')
plt.ylabel('Frequency')
plt.show()

all_stopwords = [word for sublist in train_df['stopwords'] for word in sublist]
stopword_counts = Counter(all_stopwords)
most_common_stopwords = stopword_counts.most_common(20)
print("20 Most Common Stopwords:")
for word, count in most_common_stopwords:
    print(f"{word}: {count}")

def generate_ngrams(text, n):
    words = word_tokenize(text.lower())
    return list(ngrams(words, n))

train_df['bigrams'] = train_df['text'].apply(lambda x: generate_ngrams(x, 2))
train_df['trigrams'] = train_df['text'].apply(lambda x: generate_ngrams(x, 3))

all_bigrams = [bigram for sublist in train_df['bigrams'] for bigram in sublist]
all_trigrams = [trigram for sublist in train_df['trigrams'] for trigram in sublist]

bigram_counts = Counter(all_bigrams)
trigram_counts = Counter(all_trigrams)

most_common_bigrams = bigram_counts.most_common(20)
most_common_trigrams = trigram_counts.most_common(20)

print("20 Most Common Bigrams:")
for bigram, count in most_common_bigrams:
    print(f"{bigram}: {count}")

print("\n20 Most Common Trigrams:")
for trigram, count in most_common_trigrams:
    print(f"{trigram}: {count}")

bigram_df = pd.DataFrame(most_common_bigrams, columns=['bigram', 'count'])
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='bigram', data=bigram_df)
plt.title('Top 20 Bigrams')
plt.xlabel('Count')
plt.ylabel('Bigram')
plt.show()

trigram_df = pd.DataFrame(most_common_trigrams, columns=['trigram', 'count'])
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='trigram', data=trigram_df)
plt.title('Top 20 Trigrams')
plt.xlabel('Count')
plt.ylabel('Trigram')
plt.show()

all_text = ' '.join(train_df['text'].tolist())
stop_words = set(stopwords.words('english'))
wordcloud = WordCloud(stopwords=stop_words, background_color='white', width=800, height=400).generate(all_text)

plt.figure(figsize=(15, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud for AG News Dataset', fontsize=20)
plt.show()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stop_words]
    return ' '.join(lemmatized_tokens)

train_df['processed_text'] = train_df['text'].apply(preprocess_text)

vectorizer = CountVectorizer(max_features=5000)
X_bow = vectorizer.fit_transform(train_df['processed_text'])
bow_df = pd.DataFrame(X_bow.toarray(), columns=vectorizer.get_feature_names_out())
print(bow_df.head())

lda = LatentDirichletAllocation(n_components=10, random_state=42)
lda.fit(X_bow)

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

no_top_words = 10
display_topics(lda, vectorizer.get_feature_names_out(), no_top_words)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

sample_size = 1000
sample_df = train_df.sample(sample_size, random_state=42)
sample_df['sentiment_polarity'] = sample_df['text'].apply(get_sentiment)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

sample_df['sentiment'] = sample_df['sentiment_polarity'].apply(classify_sentiment)
print(sample_df['sentiment'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=sample_df, palette='viridis')
plt.title('Sentiment Distribution in AG News Articles')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

def preprocess_text_spacy(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ' '.join(lemmatized_tokens)

train_df['processed_text'] = train_df['text'].apply(preprocess_text_spacy)

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

sample_size = 1000
sample_df = train_df.sample(sample_size, random_state=42)
sample_df['sentiment_polarity'] = sample_df['processed_text'].apply(get_sentiment)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

sample_df['sentiment'] = sample_df['sentiment_polarity'].apply(classify_sentiment)
print(sample_df['sentiment'].value_counts())

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment', data=sample_df, palette='viridis')
plt.title('Sentiment Distribution in AG News Articles')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
"""