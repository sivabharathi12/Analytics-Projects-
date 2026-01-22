import pandas as pd
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"


# load the dataset
df = pd.read_csv("chatgpt_reviews.csv")

# Wrap these in print() to see them in the terminal
print(df.head())

print(df.isnull().sum())

# ... imports ...

# load the dataset
df = pd.read_csv("chatgpt_reviews.csv")

# ===> ADD YOUR LINE HERE <===
# Convert all reviews to strings and replace missing values with empty strings
df['Review'] = df['Review'].astype(str).fillna('') 

# Check for nulls (this should now show 0 for the Review column)
print(df.isnull().sum())

from textblob import TextBlob

# function to determine sentiment polarity
def get_sentiment(review):
    sentiment = TextBlob(review).sentiment.polarity
    if sentiment > 0:
        return 'Positive'
    elif sentiment < 0:
        return 'Negative'
    else:
        return 'Neutral'

# apply sentiment analysis
df['Sentiment'] = df['Review'].apply(get_sentiment)

# Calculate the counts
sentiment_distribution = df['Sentiment'].value_counts()



fig = go.Figure(data=[go.Bar(
    x=sentiment_distribution.index,
    y=sentiment_distribution.values,
    marker_color=['green', 'gray', 'red'],
)])

fig.update_layout(
    title='Sentiment Distribution of ChatGPT Reviews',
    xaxis_title='Sentiment',
    yaxis_title='Number of Reviews',
    width=800,
    height=600
)

fig.show()


# filter reviews with positive sentiment
positive_reviews = df[df['Sentiment'] == 'Positive']['Review']

# use CountVectorizer to extract common phrases (n-grams)
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english', max_features=100)
X = vectorizer.fit_transform(positive_reviews)

# sum the counts of each phrase
phrase_counts = X.sum(axis=0)
phrases = vectorizer.get_feature_names_out()
phrase_freq = [(phrases[i], phrase_counts[0, i]) for i in range(len(phrases))]

# sort phrases by frequency
phrase_freq = sorted(phrase_freq, key=lambda x: x[1], reverse=True)

phrase_df = pd.DataFrame(phrase_freq, columns=['Phrase', 'Frequency'])

fig = px.bar(phrase_df,
             x='Frequency',
             y='Phrase',
             orientation='h',
             title='Top Common Phrases in Positive Reviews',
             labels={'Phrase': 'Phrase', 'Frequency': 'Frequency'},
             width=1000,
             height=600)

fig.update_layout(
    xaxis_title='Frequency',
    yaxis_title='Phrase',
    yaxis={'categoryorder':'total ascending'}
)

fig.show()

# filter reviews with negative sentiment
negative_reviews = df[df['Sentiment'] == 'Negative']['Review']

# use CountVectorizer to extract common phrases (n-grams) for negative reviews
X_neg = vectorizer.fit_transform(negative_reviews)

# sum the counts of each phrase in negative reviews
phrase_counts_neg = X_neg.sum(axis=0)
phrases_neg = vectorizer.get_feature_names_out()
phrase_freq_neg = [(phrases_neg[i], phrase_counts_neg[0, i]) for i in range(len(phrases_neg))]

# sort phrases by frequency
phrase_freq_neg = sorted(phrase_freq_neg, key=lambda x: x[1], reverse=True)

phrase_neg_df = pd.DataFrame(phrase_freq_neg, columns=['Phrase', 'Frequency'])

fig = px.bar(phrase_neg_df,
             x='Frequency',
             y='Phrase',
             orientation='h',
             title='Top Common Phrases in Negative Reviews',
             labels={'Phrase': 'Phrase', 'Frequency': 'Frequency'},
             width=1000,
             height=600)

fig.update_layout(
    xaxis_title='Frequency',
    yaxis_title='Phrase',
    yaxis={'categoryorder':'total ascending'}
)

fig.show()

# grouping similar phrases into broader problem categories
problem_keywords = {
    'Incorrect Answers': ['wrong answer', 'gives wrong', 'incorrect', 'inaccurate', 'wrong'],
    'App Performance': ['slow', 'lag', 'crash', 'bug', 'freeze', 'loading', 'glitch', 'worst app', 'bad app', 'horrible', 'terrible'],
    'User Interface': ['interface', 'UI', 'difficult to use', 'confusing', 'layout'],
    'Features Missing/Not Working': ['feature missing', 'not working', 'missing', 'broken', 'not available'],
    'Quality of Responses': ['bad response', 'useless', 'poor quality', 'irrelevant', 'nonsense']
}

# initialize a dictionary to count problems
problem_counts = {key: 0 for key in problem_keywords.keys()}

# count occurrences of problem-related phrases in negative reviews
for phrase, count in phrase_freq_neg:
    for problem, keywords in problem_keywords.items():
        if any(keyword in phrase for keyword in keywords):
            problem_counts[problem] += count
            break

problem_df = pd.DataFrame(list(problem_counts.items()), columns=['Problem', 'Frequency'])

fig = px.bar(problem_df,
             x='Frequency',
             y='Problem',
             orientation='h', 
             title='Common Problems Faced by Users in ChatGPT',
             labels={'Problem': 'Problem', 'Frequency': 'Frequency'},
             width=1000,
             height=600)

fig.update_layout(
    plot_bgcolor='white',  
    paper_bgcolor='white', 
    xaxis_title='Frequency',
    yaxis_title='Problem',
    yaxis={'categoryorder':'total ascending'}  
)

fig.show()


# convert 'Review Date' to datetime format
df['Review Date'] = pd.to_datetime(df['Review Date'])

# aggregate sentiment counts by date
sentiment_over_time = df.groupby([df['Review Date'].dt.to_period('M'), 'Sentiment']).size().unstack(fill_value=0)

# convert the period back to datetime for plotting
sentiment_over_time.index = sentiment_over_time.index.to_timestamp()

fig = go.Figure()

fig.add_trace(go.Scatter(x=sentiment_over_time.index, y=sentiment_over_time['Positive'],
                         mode='lines', name='Positive', line=dict(color='green')))
fig.add_trace(go.Scatter(x=sentiment_over_time.index, y=sentiment_over_time['Neutral'],
                         mode='lines', name='Neutral', line=dict(color='gray')))
fig.add_trace(go.Scatter(x=sentiment_over_time.index, y=sentiment_over_time['Negative'],
                         mode='lines', name='Negative', line=dict(color='red')))

fig.update_layout(
    title='Sentiment Trends Over Time',
    xaxis_title='Date',
    yaxis_title='Number of Reviews',
    plot_bgcolor='white',  
    paper_bgcolor='white',  
    legend_title_text='Sentiment',
    xaxis=dict(showgrid=True, gridcolor='lightgray'), 
    yaxis=dict(showgrid=True, gridcolor='lightgray')
)

fig.show()

# define the categories based on the ratings
df['NPS Category'] = df['Ratings'].apply(lambda x: 'Promoter' if x == 5 else ('Passive' if x == 4 else 'Detractor'))

# calculate the percentage of each category
nps_counts = df['NPS Category'].value_counts(normalize=True) * 100

# calculate NPS
nps_score = nps_counts.get('Promoter', 0) - nps_counts.get('Detractor', 0)

# display the NPS Score
nps_score
# ... previous code ...

# calculate the percentage of each category
nps_counts = df['NPS Category'].value_counts(normalize=True) * 100

# ===> ADD THIS LINE <===
print("NPS Category Distribution:")
print(nps_counts)

# Create a Pie Chart for NPS Distribution
fig = px.pie(

    values=nps_counts.values, 
    names=nps_counts.index, 
    title='Net Promoter Score (NPS) Category Distribution',
    template='plotly_white'
)

fig.show()