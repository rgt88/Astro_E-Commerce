import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import joblib
from scipy import sparse
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from string import punctuation

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the TF-IDF Vectorizer and Matrix
tf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = sparse.load_npz('tfidf_matrix.npz')

# Initialize BigQuery client
def get_bq_data():
    key_path = "astro-435108-c9a82eba7e13.json"
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    query = """
    WITH product_recommendation AS (
        SELECT ori.*, inv.* EXCEPT(product_id, id, created_at)
        FROM bigquery-public-data.thelook_ecommerce.order_items ori
        JOIN bigquery-public-data.thelook_ecommerce.inventory_items inv
        ON ori.inventory_item_id = inv.id
    ),
    test AS (
        SELECT order_id, user_id, product_id, created_at, inventory_item_id,
            sale_price, product_category, product_name, product_brand, product_department, status
        FROM product_recommendation
        WHERE status != 'Cancelled'
    )
    SELECT DISTINCT product_category, product_name, product_brand, product_department
    FROM product_recommendation;
    """
    df = client.query(query).to_dataframe()
    return df

# Data Preprocessing
def to_lower(text):
    return text.lower()

def remove_contraction(text):
    product_abbreviation_dict = {
    # Sleep & Lounge
    "pjs": "pajamas",
    "sleepwear": "sleep outfit",
    "nightgown": "night dress",
    "nightshirt": "sleep shirt",
    "loungewear": "lounge outfit",
    "robe": "bathrobe",
    "slippers": "house slippers",
    "sleep pants": "sleeping pants",
    "sleep tee": "sleeping t-shirt",
    "sweatpants": "joggers",
    "sleep shorts": "sleeping shorts",

    # Socks
    "socks": "foot socks",
    "ankle socks": "low-cut socks",
    "crew socks": "mid-calf socks",
    "no-show socks": "hidden socks",
    "knee-high socks": "knee-length socks",
    "compression socks": "pressure socks",

    # Socks & Hosiery
    "hosiery": "legwear",
    "stockings": "leg stockings",
    "tights": "full-length tights",
    "leggings": "footless tights",
    "pantyhose": "sheer tights",
    "thigh-high": "thigh-high stockings",

    # Suits
    "suit": "formal suit",
    "blzr": "blazer",
    "vest": "waistcoat",
    "tux": "tuxedo",
    "suit jacket": "formal jacket",
    "dress pants": "formal pants",
    "trouser": "suit trousers",
    "tailcoat": "formal tailcoat",

    # Suits & Sport Coats
    "sport coat": "sports jacket",
    "casual blazer": "informal blazer",
    "double-breasted": "double-breasted jacket",
    "single-breasted": "single-breasted jacket",
    "dinner jacket": "evening jacket",

    # Sweaters
    "sweater": "knit sweater",
    "turtleneck": "turtleneck sweater",
    "cardi": "cardigan",
    "pullover": "pullover sweater",
    "crewneck": "crewneck sweater",
    "v-neck": "v-neck sweater",
    "sweatshirt": "casual sweater",

    # Swim
    "swim trunks": "swimming shorts",
    "bikini": "two-piece swimsuit",
    "one-piece": "one-piece swimsuit",
    "swim top": "bikini top",
    "board shorts": "surfing shorts",
    "rashguard": "swim shirt",
    "tankini": "tank top swimsuit",

    # Tops & Tees
    "tee": "t-shirt",
    "polo": "polo shirt",
    "blouse": "casual top",
    "tank top": "sleeveless top",
    "crop top": "cropped t-shirt",
    "v-neck tee": "v-neck t-shirt",
    "henley": "collarless shirt",
    "button-up": "button-up shirt",
    "long-sleeve tee": "long-sleeve t-shirt",

    # Underwear
    "boxers": "boxer shorts",
    "briefs": "underwear briefs",
    "trunks": "short briefs",
    "bikini briefs": "low-rise briefs",
    "boxer briefs": "fitted boxer briefs",
    "thong": "thong underwear",
    "panties": "women's underwear",
    "bralette": "light bra",
    "sports bra": "athletic bra",
    "undershirt": "base layer shirt"
}
    list_kata = text.split()
    list_hasil = [product_abbreviation_dict.get(kata, kata) for kata in list_kata]
    return ' '.join(list_hasil)

def remove_number(text):
    return ''.join([char for char in text if not char.isnumeric()])

def remove_punctuation(text):
    return ''.join([char for char in text if char not in punctuation])

def remove_stopwords(text):
    stop_words = stopwords.words('english')
    return ' '.join([kata for kata in text.split() if kata not in stop_words])

def remove_whitespace(text):
    return ' '.join(text.split())

def stem(text):
    stemmer = SnowballStemmer('english')
    return ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(text)])

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text)])

def combine_cleaning(series):
    df['text_clean'] = series.apply(to_lower)
    df['text_clean'] = df['text_clean'].apply(remove_contraction)
    df['text_clean'] = df['text_clean'].apply(remove_number)
    df['text_clean'] = df['text_clean'].apply(remove_punctuation)
    df['text_clean'] = df['text_clean'].apply(remove_stopwords)
    df['text_clean'] = df['text_clean'].apply(remove_whitespace)
    df['stem'] = df['text_clean'].apply(stem)
    df['lemmatize'] = df['text_clean'].apply(lemmatize)
    return df

def recommend_products(input_product_name, df, cosine_sim):
    closest_match = process.extractOne(input_product_name, df['lemmatize'].tolist())
    if closest_match:
        product_name = closest_match[0]
        idx = df[df['lemmatize'] == product_name].index[0]
        sim_scores = list(enumerate(cosine_similarity(tfidf_matrix)[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4]
        product_indices = [i[0] for i in sim_scores]
        return df.iloc[product_indices][['product_category', 'product_name', 'product_brand', 'product_department']]
    else:
        return "No matching product found."

# Streamlit App
st.title('Product Recommendation System')

# Data Retrieval and Display
df = get_bq_data()
df['gabungan'] = df['product_category'] + ' ' + df['product_name'] + ' ' + df['product_brand'] + ' ' + df['product_department']
df = combine_cleaning(df['gabungan'])
tfidf_matrix = tf.fit_transform(df['gabungan'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

st.write("### Data Overview")
st.dataframe(df)

# EDA Plots
st.write("### Total Product by Department")
product_gender = df.groupby('product_department').size().reset_index(name='count').sort_values(by='count', ascending=False)
product_gender['percentage'] = (product_gender['count'] / product_gender['count'].sum()) * 100
fig, ax = plt.subplots()
ax.pie(product_gender['count'], labels=product_gender['product_department'], startangle=90, counterclock=False, explode=[0.1, 0.0],
       autopct='%.f%%', colors=['red', 'green'])
ax.set_title('Total Product by Department')
st.pyplot(fig)

st.write("### Top 10 Product Categories")
product_category = df.groupby('product_category').size().reset_index(name='count').sort_values(by='count', ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
clrs = ['grey' if (x < max(product_category['count'])) else 'red' for x in product_category['count']]
sns.barplot(data=product_category.head(10), y='product_category', x='count', palette=clrs, ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_width()} Products", xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(-70, 0), textcoords='offset points', ha="left", va="center", color='white')
ax.set_title('Top 10 Product Categories')
st.pyplot(fig)

st.write("### Top 10 Product Brands")
product_brand = df.groupby('product_brand').size().reset_index(name='count').sort_values(by='count', ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
clrs = ['grey' if (x < max(product_brand['count'])) else 'red' for x in product_brand['count']]
sns.barplot(data=product_brand.head(10), y='product_brand', x='count', palette=clrs, ax=ax)
for p in ax.patches:
    ax.annotate(f"{p.get_width()} Products", xy=(p.get_width(), p.get_y() + p.get_height() / 2),
                xytext=(-70, 0), textcoords='offset points', ha="left", va="center", color='white')
ax.set_title('Top 10 Product Brands')
st.pyplot(fig)

# User Input and Recommendations
user_input = st.text_input('Enter product name to get recommendations:')
if user_input:
    recommendations = recommend_products(user_input, df, cosine_sim)
    st.write('### Recommendations:')
    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        st.dataframe(recommendations)

# Save and Load models (example usage)
# joblib.dump(tf, 'tfidf_vectorizer.pkl')
# sparse.save_npz('tfidf_matrix.npz', tfidf_matrix)
