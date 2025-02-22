import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st

# File paths
DATA_PATH = r"C:\Users\WINDOWS\Desktop\pro4 datas\swiggy_data.csv"
CLEANED_DATA_PATH = r"C:\Users\WINDOWS\Desktop\New folder\cleaned_data.csv"
ENCODED_DATA_PATH = r"C:\Users\WINDOWS\Desktop\New folder\encoded_data.csv"
ENCODER_PATH = r"C:\Users\WINDOWS\Desktop\New folder\encoder.pkl"



# ------------------ DATA CLEANING ------------------
def clean_data(file_path):
    df = pd.read_csv(file_path).head(10000)
    df.drop_duplicates(inplace=True)    
    
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    def convert_rating_count(value):

        if pd.isna(value) or value == 'Too Few Ratings':
            return 0  # or np.nan depending on how you want to handle it
        elif 'K' in value:
            return int(value.replace('K+ ratings', '').replace('K', '').strip()) * 1000
        elif '+' in value:
            return int(value.replace('+ ratings', '').strip())
        else:
            return 0  # For any non-standard value, you can handle it as needed

# Apply the function to the 'rating_count' column
    df['rating_count'] = df['rating_count'].apply(convert_rating_count)
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['cost'] = df['cost'].replace({'₹': ''}, regex=True).astype(float)
    #df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
    df['rating'] = df['rating'].replace('--', np.nan)
    df['rating'].fillna(df['rating'].median(), inplace=True)
    df['rating_count'].fillna(df['rating_count'].median(), inplace=True)
    

    for col in ['name', 'city', 'cuisine', 'address']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df.drop(columns=['id', 'lic_no', 'link', 'menu'], errors='ignore', inplace=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    return df

# ------------------ DATA PREPROCESSING ------------------
def preprocess_data(df):
    encoder = OneHotEncoder(handle_unknown='ignore')
    categorical_features = ['name', 'city', 'cuisine']

    df = df.dropna(subset=categorical_features)
    encoded_features = encoder.fit_transform(df[categorical_features])

    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(categorical_features))

    numerical_features = df[['rating', 'rating_count', 'cost']].reset_index(drop=True)
    final_df = pd.concat([numerical_features, encoded_df], axis=1)

    final_df.fillna(0, inplace=True)

    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder, f)

    final_df.to_csv(ENCODED_DATA_PATH, index=False)
    return final_df

# ------------------ RECOMMENDATION FUNCTION ------------------
def get_recommendations(user_vector, df, cleaned_df):
    df.fillna(0, inplace=True)

    if user_vector.isna().all().all():
        return pd.DataFrame()

    # Compute similarity between user input and all restaurants
    user_sparse = csr_matrix(user_vector)
    df_sparse = csr_matrix(df)

    similarity_scores = cosine_similarity(user_sparse, df_sparse).flatten()
    
    # Get top 5 recommendations
    top_indices = similarity_scores.argsort()[::-1][:5]

    return cleaned_df.iloc[top_indices]

# ------------------ STREAMLIT APPLICATION ------------------
def main():
    st.title("🍽️ Restaurant Recommendation System")

    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    encoded_df = pd.read_csv(ENCODED_DATA_PATH)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    cleaned_df.dropna(subset=['name', 'city', 'cuisine'], inplace=True)

    city = st.selectbox("🏙️ Select City", cleaned_df['city'].unique())
    cuisine = st.selectbox("🍽️ Select Cuisine", cleaned_df['cuisine'].unique())
    name = st.selectbox("📍 Select Restaurant", cleaned_df['name'].unique())  

    rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0)

    # Ensure 'cost' is numeric before finding max/median
    cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
    cleaned_df['cost'].fillna(cleaned_df['cost'].median(), inplace=True)  # ✅ Fix None values

    max_cost = int(cleaned_df['cost'].max(skipna=True)) if not cleaned_df['cost'].isna().all() else 1000
    median_cost = int(cleaned_df['cost'].median(skipna=True)) if not cleaned_df['cost'].isna().all() else 500

    cost = st.slider("💰 Maximum Cost", 0, max_cost, median_cost)

    user_input_df = pd.DataFrame([[name, city, cuisine]], columns=['name', 'city', 'cuisine'])

    try:
        user_encoded = encoder.transform(user_input_df)
        user_vector = pd.DataFrame(user_encoded.toarray(), columns=encoder.get_feature_names_out(['name', 'city', 'cuisine']))
    except ValueError:
        st.error("⚠️ Feature names do not match! Try selecting different inputs.")
        return

    # Ensure numerical columns are in user_vector
    user_vector[['rating', 'rating_count', 'cost']] = [[rating, 0, cost]]

    recommendations = get_recommendations(user_vector, encoded_df, cleaned_df)

    if not recommendations.empty:
        st.write("### 🍽️ Recommended Restaurants")
        st.dataframe(recommendations[['name', 'city', 'rating', 'cost', 'cuisine']], width=1000)
    else:
        st.write("❌ No matching restaurants found. Try different filters.")

if __name__ == "__main__":
    df = clean_data(DATA_PATH)
    encoded_df = preprocess_data(df)
    main()
