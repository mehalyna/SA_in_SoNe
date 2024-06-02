import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import seaborn as sns

nltk.download('stopwords')

# Step 1: Read Data from CSV
def read_csv(file):
    df = pd.read_csv(file)
    st.write("Columns in the uploaded file:", df.columns.tolist())
    
    # Try to find a column that contains 'Text' or similar
    text_column = None
    for col in df.columns:
        if 'text' in col.lower():
            text_column = col
            break
    
    if text_column is None:
        st.error("No column found that contains feedback text.")
        return [], None
    
    feedback_texts = df[text_column].dropna().tolist()
    return feedback_texts, df

# Step 2: Prepare Data
def prepare_data(texts):
    stop_words = set(stopwords.words('english'))
    texts = ' '.join(texts).lower()
    tokens = nltk.word_tokenize(texts)
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Step 3: Identify Most Frequently Used Terms
def get_most_frequent_terms(tokens, top_n=20):
    counter = Counter(tokens)
    return counter.most_common(top_n)

# Step 4: Create Document Term Matrix
def create_document_term_matrix(tokens):
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([' '.join(tokens)])
    return dtm, vectorizer

# Step 5: Topic Modeling with LDA
def perform_lda(dtm, vectorizer, n_topics=3):
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(dtm)
    return lda, vectorizer

# Step 6: Visualize Results - Word Cloud
def generate_wordcloud(tokens):
    text = ' '.join(tokens)
    wordcloud = WordCloud(stopwords=stopwords.words('english')).generate(text)
    return wordcloud

# Step 7: Social Network Graphing
def create_social_network(tokens):
    G = nx.Graph()
    G.add_nodes_from(tokens)
    edges = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    G.add_edges_from(edges)
    return G

# Step 8: Filter Mood-related Topics
def filter_mood_topics(lda, vectorizer, mood_keywords):
    mood_topics = []
    for idx, topic in enumerate(lda.components_):
        topic_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
        if any(term in mood_keywords for term in topic_terms):
            mood_topics.append((f'Topic {idx}', topic_terms))
    return mood_topics

# Streamlit UI
st.title('Customer Satisfaction Analysis using CSV Feedback Data')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file:
    with st.spinner('Reading data...'):
        feedback_texts, df = read_csv(uploaded_file)
        if feedback_texts:
            with st.spinner('Preparing data...'):
                tokens = prepare_data(feedback_texts)
            with st.spinner('Identifying most frequent terms...'):
                freq_terms = get_most_frequent_terms(tokens)
                st.write('Most Frequent Terms:', freq_terms)
                
                # Bar chart of most frequent terms
                terms_df = pd.DataFrame(freq_terms, columns=['Term', 'Count'])
                plt.figure(figsize=(10, 5))
                sns.barplot(x='Term', y='Count', data=terms_df)
                plt.xticks(rotation=45)
                plt.title('Top 20 Most Frequent Terms')
                st.pyplot(plt)
                
            with st.spinner('Creating document term matrix...'):
                dtm, vectorizer = create_document_term_matrix(tokens)
            with st.spinner('Performing LDA...'):
                lda, vectorizer = perform_lda(dtm, vectorizer)
            with st.spinner('Generating word cloud...'):
                wordcloud = generate_wordcloud(tokens)
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

            # Define mood-related keywords
            mood_keywords = ['happy', 'sad', 'angry', 'joyful', 'frustrated', 'pleased', 'disappointed'\
            , 'satisfied', 'unsatisfied', 'positive', 'negative', 'awful', 'terrible']

            # Display mood-related topics
            mood_topics = filter_mood_topics(lda, vectorizer, mood_keywords)
            st.write('Mood-related Topics identified:')
            for topic_name, topic_terms in mood_topics:
                st.write(f'{topic_name}:', topic_terms)
                
                # Bar chart of top terms in mood-related topics
                terms_df = pd.DataFrame(topic_terms, columns=['Term'])
                terms_df['Count'] = range(len(topic_terms), 0, -1)
                plt.figure(figsize=(10, 5))
                sns.barplot(x='Term', y='Count', data=terms_df)
                plt.xticks(rotation=45)
                plt.title(f'Top Terms in {topic_name}')
                st.pyplot(plt)

                # Word cloud for each mood-related topic
                wordcloud = generate_wordcloud(topic_terms)
                plt.figure(figsize=(8, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {topic_name}')
                st.pyplot(plt)

            # Display all topics and generate visualizations
            st.write('All Topics identified:')
            for idx, topic in enumerate(lda.components_):
                topic_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[:-11:-1]]
                st.write(f'Topic {idx}:', topic_terms)
                
                # Bar chart of top terms in each topic
                terms_df = pd.DataFrame(topic_terms, columns=['Term'])
                terms_df['Count'] = range(len(topic_terms), 0, -1)
                plt.figure(figsize=(10, 5))
                sns.barplot(x='Term', y='Count', data=terms_df)
                plt.xticks(rotation=45)
                plt.title(f'Top Terms in Topic {idx}')
                st.pyplot(plt)

                # Word cloud for each topic
                wordcloud = generate_wordcloud(topic_terms)
                plt.figure(figsize=(8, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for Topic {idx}')
                st.pyplot(plt)

                # Social network graph for each topic
                G = create_social_network(topic_terms)
                plt.figure(figsize=(8, 8))
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, edge_color='gray')
                plt.title(f'Social Network Graph for Topic {idx}')
                st.pyplot(plt)

            # Pie chart of sentiments
            if 'Sentiment' in df.columns:
                sentiment_counts = df['Sentiment'].value_counts()
                plt.figure(figsize=(8, 8))
                plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
                plt.title('Sentiment Distribution')
                st.pyplot(plt)

            # Time series analysis if 'Date/Time' column is available
            if 'Date/Time' in df.columns:
                df['Date/Time'] = pd.to_datetime(df['Date/Time'], errors='coerce')
                df = df.dropna(subset=['Date/Time'])
                df.set_index('Date/Time', inplace=True)
                df['Count'] = 1
                time_series = df['Count'].resample('M').sum()

                plt.figure(figsize=(10, 5))
                plt.plot(time_series.index, time_series.values, marker='o')
                plt.title('Feedback Frequency Over Time')
                plt.xlabel('Date')
                plt.ylabel('Number of Feedbacks')
                st.pyplot(plt)
