from flask import Flask, render_template, request, redirect, url_for, flash
from recommendation import get_recommendations, df, indices

app = Flask(__name__)

app.secret_key = 'your_secret_key'

@app.route('/')
def home():
    return render_template('index.html', movies=list(df['tmdb_title'][:10]))

@app.route('/recommendations', methods=['POST'])
def recommendations():
    title = request.form.get('title')
    
    if not title:
        flash("Please enter a movie title before submitting.", "warning")
        return redirect(url_for('home'))
    
    try:
        _ = indices[title]
    except:
        flash("Please enter a valid movie title.", "warning")
        return redirect(url_for('home'))
    
    factors = {
        'overview': float(request.form.get('overview', 0)),
        'reviews': float(request.form.get('reviews', 0)),
        'imdb_score': float(request.form.get('imdb_score', 0)),
        'tomatometer': float(request.form.get('tomatometer', 0)),
        'keywords': float(request.form.get('keywords', 0)),
        'genres': float(request.form.get('genres', 0)),
        'country': float(request.form.get('country', 0)),
        'length': float(request.form.get('length', 0)),
        'revenue': float(request.form.get('revenue', 0)),
        'year': float(request.form.get('year', 0)),
        'language': float(request.form.get('language', 0)),
    }
    idx, recommended_movies = get_recommendations(title, factors)
    return render_template(
        'recommendations.html',
        input_movie=df.iloc[idx].to_dict(),
        recommended_movies=recommended_movies.to_dict(orient='records'),
        movie_indices=recommended_movies.index,
        input_movie_index=idx,
        zip=zip,
        enumerate=enumerate,
    )

@app.route('/all_movies')
def all_movies():
    return render_template(
        'all_movies.html',
        movies=df.to_dict(orient='records'),
        enumerate=enumerate,
    )

@app.route('/random_movies')
def random_movies():
    num_movies = 10  # Number of random movies to select
    random_indices = df.sample(num_movies).index
    random_movies = df.iloc[random_indices].to_dict(orient='records')
    return render_template(
        'random_movies.html',
        movies=random_movies,
        movie_indices=random_indices,
        zip=zip,
    )

if __name__ == '__main__':
    app.run(host='::', port=5000, debug=True)