from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import joblib
import random
import os
import tmdbsimple as tmdb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend requests

# Configure TMDB API Key
tmdb.API_KEY = os.getenv('TMDB_API_KEY')
if not tmdb.API_KEY:
    print("WARNING: TMDB_API_KEY not found in environment. Poster fetching will fail.")

# Define paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'moviesDb.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'logistic_regression_model.joblib')
COLUMNS_PATH = os.path.join(BASE_DIR, 'models', 'model_columns.joblib')

# Load data and model globally
print(f"Loading data from {DATA_PATH}...")
try:
    movies_df = pd.read_csv(DATA_PATH)
    # Convert relevant columns to string to avoid issues with mixed types if any, and handle NaN for filters
    for col in ['genre', 'country', 'certification_US']:
        if col in movies_df.columns:
            movies_df[col] = movies_df[col].astype(str).fillna('Unknown') 
        else:
            print(f"Warning: Filter column '{col}' not found in moviesDb.csv. Filtering by it will not work.")
except FileNotFoundError:
    print(f"FATAL ERROR: Data file not found at {DATA_PATH}")
    movies_df = pd.DataFrame()

print(f"Loading model from {MODEL_PATH}...")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    model = None

print(f"Loading model columns from {COLUMNS_PATH}...")
try:
    model_columns = joblib.load(COLUMNS_PATH)
except FileNotFoundError:
    print(f"FATAL ERROR: Model columns file not found at {COLUMNS_PATH}")
    model_columns = []


@app.route('/api/quiz/filter-options', methods=['GET'])
def get_filter_options():
    if movies_df.empty:
        return jsonify({"error": "Movie data not available"}), 500
    
    options = {}
    try:
        # For genre, country, certification_US, get unique non-null, sorted values
        # The .astype(str).fillna('Unknown') during load should prevent actual NaNs here
        if 'genre' in movies_df.columns:
            options['genres'] = sorted([g for g in movies_df['genre'].unique() if g and g != 'nan' and g != 'Unknown'])
        else:
            options['genres'] = []

        if 'country' in movies_df.columns:
            options['countries'] = sorted([c for c in movies_df['country'].unique() if c and c != 'nan' and c != 'Unknown'])
        else:
            options['countries'] = []

        if 'certification_US' in movies_df.columns:
            options['certifications'] = sorted([r for r in movies_df['certification_US'].unique() if r and r != 'nan' and r != 'Unknown'])
        else:
            options['certifications'] = []
        
        return jsonify(options)
    except Exception as e:
        print(f"Error fetching filter options: {e}")
        return jsonify({"error": "Could not retrieve filter options"}), 500


@app.route('/api/quiz/next-movie', methods=['GET'])
def get_next_movie():
    if movies_df.empty:
        return jsonify({"error": "Movie data not available"}), 500

    filtered_df = movies_df.copy()

    # Get filter parameters from request query string
    genre_filter = request.args.get('genre')
    country_filter = request.args.get('country')
    certification_filter = request.args.get('certification') # Match query param name

    if genre_filter and 'genre' in filtered_df.columns:
        # Make comparison case-insensitive and strip whitespace
        search_genre = genre_filter.strip().lower()
        filtered_df = filtered_df[filtered_df['genre'].str.strip().str.lower() == search_genre]
    
    if country_filter and 'country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['country'] == country_filter]

    if certification_filter and 'certification_US' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['certification_US'] == certification_filter]

    if filtered_df.empty:
        return jsonify({"error": "No movies found matching your criteria. Try broadening your filters!"}), 404 # 404 Not Found

    random_movie_from_db = filtered_df.sample(1).iloc[0]
    movie_id = int(random_movie_from_db['id'])
    movie_title = random_movie_from_db['title']
    poster_path = None # Default to None

    if tmdb.API_KEY:
        try:
            movie_tmdb = tmdb.Movies(movie_id)
            response = movie_tmdb.info()
            poster_path = response.get('poster_path', None)
            # Potentially update title if it's more current/accurate from TMDB, though be careful with consistency
            # movie_title = response.get('title', movie_title) 
        except Exception as e:
            print(f"Error fetching movie details (ID: {movie_id}) from TMDB: {e}")
            # Proceed with poster_path as None, use title from CSV
    
    return jsonify({
        "id": str(movie_id), 
        "title": movie_title,
        "posterUrl": poster_path # This is the path, e.g., /path.jpg or None
    })


@app.route('/api/quiz/submit-guess', methods=['POST'])
def submit_guess():
    if model is None or not model_columns or movies_df.empty:
        return jsonify({"error": "Backend model or data not properly loaded"}), 500

    data = request.get_json()
    movie_id_str = data.get('movieId')
    user_guess_str = data.get('guess') # 'Hit' or 'Flop'

    if not movie_id_str or not user_guess_str:
        return jsonify({"error": "Missing movieId or guess in request"}), 400
    
    try:
        movie_id = int(movie_id_str.replace('tmdb-', '')) 
    except ValueError:
        try:
            movie_id = int(movie_id_str)
        except ValueError:
             return jsonify({"error": "Invalid movieId format"}), 400

    movie_data_from_db = movies_df[movies_df['id'] == movie_id]

    if movie_data_from_db.empty:
        return jsonify({"error": f"Movie with ID {movie_id} not found in local DB"}), 404

    single_movie_features = movie_data_from_db.copy()

    for col in ['certification_US', 'genre', 'country']:
        if col in single_movie_features.columns and single_movie_features[col].isnull().any():
            # Ensure NaNs are handled as 'Unknown' for consistency with training
            single_movie_features.loc[:, col] = single_movie_features[col].fillna('Unknown')
    
    predictors = ['budget', 'runtime', 'year', 'vote_average', 'vote_count', 'certification_US', 'genre', 'country']
    
    # Ensure the columns exist and handle potential missing ones for robustness before get_dummies
    # This part can be tricky if the CSV doesn't always have all predictors for a selected movie
    for p in predictors:
        if p not in single_movie_features.columns:
            # Decide a strategy: fill with 0/mean for numeric, or a placeholder for categorical for dummification
            # For now, if a predictor column is entirely missing (should not happen if CSV schema is consistent)
            # this will error out at pd.get_dummies or reindex. A more robust API would handle this.
            print(f"Warning: Predictor column '{p}' missing for movie ID {movie_id}. Prediction might be affected.")
            # If it's a categorical column that might be dummified, add it with a placeholder
            if p in ['certification_US', 'genre', 'country']:
                single_movie_features[p] = 'Unknown' 
            # If numeric, this might need a 0 or mean, but model_columns reindex handles missing *dummy* columns

    single_movie_raw_features = single_movie_features[predictors]

    try:
        processed_movie = pd.get_dummies(single_movie_raw_features, columns=['certification_US', 'genre', 'country'])
        processed_movie = processed_movie.reindex(columns=model_columns, fill_value=0)
    except Exception as e:
        print(f"Error during dummification/reindexing for movie ID {movie_id}: {e}")
        return jsonify({"error": "Error processing movie features for prediction"}), 500

    prediction_proba = model.predict_proba(processed_movie)[0] 
    model_prediction_numeric = model.predict(processed_movie)[0] 
    model_prediction_str = 'Hit' if model_prediction_numeric else 'Flop'

    actual_success_bool = movie_data_from_db['success'].iloc[0]
    actual_result_str = 'Hit' if actual_success_bool else 'Flop'
    
    is_correct = (user_guess_str == actual_result_str)

    feedback_message = f"You guessed {user_guess_str}. "
    if is_correct:
        feedback_message += f"Correct! It was indeed a {actual_result_str}."
    else:
        feedback_message += f"Not quite! It was actually a {actual_result_str}."
    feedback_message += f" (Model predicted: {model_prediction_str} with {prediction_proba[1]*100:.0f}% confidence for Hit)"

    return jsonify({
        "movieId": movie_id_str,
        "userGuess": user_guess_str,
        "prediction": model_prediction_str,
        "actualResult": actual_result_str,
        "isCorrect": is_correct,
        "feedbackMessage": feedback_message
    })

if __name__ == '__main__':
    # Ensure the app runs on a port different from Vite default if run on same machine
    app.run(debug=True, port=5001) # Using port 5001 to avoid Vite's 5173
