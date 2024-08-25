from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load and preprocess the dataset
file_path = 'My Projects/BR3/ld.csv'
books_df = pd.read_csv(file_path)

books_df['Author(s)'].fillna('', inplace=True)
books_df['Description'].fillna('', inplace=True)
books_df['Publisher(s)'].fillna('', inplace=True)

books_df['combined_features'] = books_df['Title'] + ' ' + \
    books_df['Author(s)'] + ' ' + books_df['Description'] + ' ' + books_df['Publisher(s)']

# Create the TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books_df['combined_features'])
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


def get_recommendations(Title, cosine_sim=cosine_sim_matrix):
    # Check if the title exists in the DataFrame
    if Title not in books_df['Title'].values:
        return []

    # Get the index of the book that matches the title
    idx = books_df[books_df['Title'] == Title].index[0]

    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return books_df['Title'].iloc[book_indices].tolist()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']
    recommendations = get_recommendations(book_title, cosine_sim_matrix)
    if not recommendations:
        error_message = f"No recommendations found for '{book_title}'. Please try another title."
        return render_template('index.html', error_message=error_message, book_title=book_title)
    return render_template('index.html', recommendations=recommendations, book_title=book_title)


if __name__ == '__main__':
    app.run(debug=True)
