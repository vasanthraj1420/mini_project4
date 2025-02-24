import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import streamlit as st

# File paths
DATA_PATH = r"C:\Users\WINDOWS\Desktop\pro4 datas\swiggy_data.csv"
CLEANED_DATA_PATH = r"C:\Users\WINDOWS\Desktop\New folder\cleaned_data.csv"
ENCODED_DATA_PATH = r"C:\Users\WINDOWS\Desktop\New folder\encoded_data.csv"
ENCODER_PATH = r"C:\Users\WINDOWS\Desktop\New folder\encoder.pkl"
KMEANS_MODEL_PATH = r"C:\Users\WINDOWS\Desktop\New folder\kmeans.pkl"

# ------------------ DATA CLEANING ------------------
def clean_data(file_path):
    df = pd.read_csv(file_path).head(10000)
    df.drop_duplicates(inplace=True)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    def convert_rating_count(value):
        if pd.isna(value) or value == 'Too Few Ratings':
            return 0  
        elif 'K' in value:
            return int(value.replace('K+ ratings', '').replace('K', '').strip()) * 1000
        elif '+' in value:
            return int(value.replace('+ ratings', '').strip())
        else:
            return 0 

    df['rating_count'] = df['rating_count'].apply(convert_rating_count)
    df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')
    df['cost'] = df['cost'].replace({'₹': ''}, regex=True).astype(float)
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

# ------------------ APPLY K-MEANS CLUSTERING ------------------
def apply_kmeans(df, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df)

    # Save K-Means model
    with open(KMEANS_MODEL_PATH, "wb") as f:
        pickle.dump(kmeans, f)

    df.to_csv(ENCODED_DATA_PATH, index=False)  # ✅ Ensure the cluster column is saved

    return df, kmeans

# ------------------ RECOMMENDATION FUNCTION ------------------
def get_recommendations(restaurant_name, df, cleaned_df):
    if 'cluster' not in df.columns:
        return "❌ Error: Clustering not applied. Run apply_kmeans() first!"

    if restaurant_name not in cleaned_df['name'].values:
        return "❌ Restaurant not found!"

    idx = cleaned_df[cleaned_df['name'] == restaurant_name].index[0]
    cluster_label = df.loc[idx, 'cluster']

    recommendations = cleaned_df[df['cluster'] == cluster_label].sample(5)

    return recommendations

# ------------------ STREAMLIT APPLICATION ------------------
def main():
    st.title("🍽️ Restaurant Recommendation System")

    # ✅ Load cleaned & encoded data
    cleaned_df = pd.read_csv(CLEANED_DATA_PATH)
    encoded_df = pd.read_csv(ENCODED_DATA_PATH)

    # ✅ Check if clustering has been applied (Apply K-Means if missing)
    if 'cluster' not in encoded_df.columns:
        st.warning("⚠️ Applying K-Means clustering...")
        encoded_df, _ = apply_kmeans(encoded_df)  # Apply clustering if not already done

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    cleaned_df.dropna(subset=['name', 'city', 'cuisine'], inplace=True)

    city = st.selectbox("🏙️ Select City", cleaned_df['city'].unique())
    cuisine = st.selectbox("🍽️ Select Cuisine", cleaned_df['cuisine'].unique())
    name = st.selectbox("📍 Select Restaurant", cleaned_df['name'].unique())  

    rating = st.slider("⭐ Minimum Rating", 0.0, 5.0, 3.0)

    cleaned_df['cost'] = pd.to_numeric(cleaned_df['cost'], errors='coerce')
    cleaned_df['cost'].fillna(cleaned_df['cost'].median(), inplace=True)

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

    user_vector[['rating', 'rating_count', 'cost']] = [[rating, 0, cost]]

    recommendations = get_recommendations(name, encoded_df, cleaned_df)

    if isinstance(recommendations, str):
        st.write(recommendations)
    else:
        recommendations['cost'] = '₹' + recommendations['cost'].astype(str)
        st.write("### 🍽️ Recommended Restaurants")
        st.dataframe(recommendations[['name', 'city', 'rating', 'cost', 'cuisine']], width=5000)

if __name__ == "__main__":
    df = clean_data(DATA_PATH)
    encoded_df = preprocess_data(df)

    # ✅ Ensure K-Means Clustering is applied **before** running the app
    if 'cluster' not in encoded_df.columns:
        encoded_df, kmeans_model = apply_kmeans(encoded_df)

    main()
