from flask import Flask, request, render_template, url_for
import math
import pandas as pd
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split

app = Flask(__name__)

# --------------------------------------------------------------------
# 1. LOAD DATA (combined_dataset.csv) AND TRAIN SVD MODEL
# --------------------------------------------------------------------
combined_df = pd.read_csv('combined_dataset.csv')

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(combined_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)
mae = accuracy.mae(predictions)
rmse = accuracy.rmse(predictions)
print("MAE:", mae, "RMSE:", rmse)

# --------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------
def get_user_recommendations(user_id, top_n=8):
    all_movies = combined_df[['movieId', 'title']].drop_duplicates()
    user_rated = combined_df[combined_df['userId'] == user_id]['movieId'].unique()
    candidates = all_movies[~all_movies['movieId'].isin(user_rated)]
    pred_list = []
    for _, row in candidates.iterrows():
        iid = row['movieId']
        pred = model.predict(user_id, iid)
        pred_list.append((iid, pred.est, row['title']))
    pred_list.sort(key=lambda x: x[1], reverse=True)
    return pred_list[:top_n]

def get_unique_genres(df):
    genres_set = set()
    for item in df['genres'].dropna():
        for g in item.split('|'):
            g_clean = g.strip()
            if g_clean and g_clean != "(no genres listed)":
                genres_set.add(g_clean)
    return sorted(genres_set)

def filter_movies_by_genre(df, genre=None):
    if not genre or genre.lower() == 'default':
        return df.head(300).copy()
    else:
        return df[df['genres'].str.contains(genre, case=False, na=False)].copy()

def get_movie_ratings(df):
    grouped = df.groupby(['movieId', 'title', 'genres'], as_index=False)['rating'].mean()
    grouped.rename(columns={'rating': 'avg_rating'}, inplace=True)
    return grouped

def create_pagination_links(base_url, current_page, total_pages):
    page_links_html = []
    if current_page > 1:
        page_links_html.append(f'<a href="{base_url}page={current_page-1}#movies" class="ajax-link">&larr; Previous</a>')
    else:
        page_links_html.append('<span style="color:#ccc;">&larr; Previous</span>')
    if total_pages <= 5:
        pages_to_show = list(range(1, total_pages+1))
    else:
        pages_to_show = [1, 2, 3, 4]
        if current_page not in pages_to_show and current_page < total_pages - 2:
            pages_to_show.append('...')
        if current_page not in [1, 2, 3, 4, total_pages] and (1 < current_page < total_pages):
            pages_to_show.append(current_page)
        if total_pages not in pages_to_show:
            pages_to_show.append('...')
            pages_to_show.append(total_pages)
        new_list = []
        for p in pages_to_show:
            if p not in new_list:
                new_list.append(p)
        pages_to_show = sorted(new_list, key=lambda x: (x != '...', x))
    for p in pages_to_show:
        if p == '...':
            page_links_html.append('<span>...</span>')
        else:
            if p == current_page:
                page_links_html.append(
                    f'<span style="background-color:#000; color:#fff; padding:4px 8px; margin:0 2px;">{p}</span>'
                )
            else:
                page_links_html.append(f'<a href="{base_url}page={p}#movies" class="ajax-link">{p}</a>')
    if current_page < total_pages:
        page_links_html.append(f'<a href="{base_url}page={current_page+1}#movies" class="ajax-link">Next &rarr;</a>')
    else:
        page_links_html.append('<span style="color:#ccc;">Next &rarr;</span>')
    return ' '.join(page_links_html)

movie_ratings_df = get_movie_ratings(combined_df)
all_genres = get_unique_genres(combined_df)
all_genres = ['Default'] + all_genres

# --------------------------------------------------------------------
# 3. ROUTES
# --------------------------------------------------------------------
@app.route('/')
def home():
    genre = request.args.get('genre', default='Default')
    base_url = f"/?genre={genre}&ajax=1&"
    filtered_movies = filter_movies_by_genre(movie_ratings_df, genre)
    filtered_movies = filtered_movies.sort_values(by='avg_rating', ascending=False)
    page = request.args.get('page', default=1, type=int)
    items_per_page = 8
    total_items = len(filtered_movies)
    total_pages = math.ceil(total_items / items_per_page)
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_movies = filtered_movies.iloc[start_idx:end_idx]
    pagination_html = create_pagination_links(base_url, page, total_pages)
    is_ajax = request.args.get('ajax') == '1'
    if is_ajax:
        return render_template('movies_fragment.html', page_movies=page_movies, pagination_html=pagination_html)
    return render_template('index.html', all_genres=all_genres, page_movies=page_movies, pagination_html=pagination_html, genre=genre)

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('user_id')
    if not user_id:
        return "User ID cannot be empty."
    try:
        user_id_int = int(user_id)
    except ValueError:
        return "Invalid User ID; it must be a number."
    recs = get_user_recommendations(user_id_int, top_n=8)
    return render_template('login.html', user_id=user_id, recs=recs)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
