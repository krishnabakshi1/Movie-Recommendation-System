import pandas as pd
import numpy as np 
import streamlit as st
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)
pd.set_option('display.max_colwidth', 500)
pd.set_option('expand_frame_repr', True)

 


movies = pd.read_csv("movies.csv")
credits1 = pd.read_csv("Credits100.csv")
credits2 = pd.read_csv("Credits200.csv")
credits3 = pd.read_csv("Credits300.csv")

credits = pd.concat([credits1,credits2,credits3])



combined_data = movies.merge(credits, on='title')


features_cols = ['id', 'title', 'tagline', 'genres', 'keywords', 'cast', 'crew', 'overview']

final_df = combined_data[features_cols]

# We select features that we can use to extract select revevant keywords or tags from. Keywords from string features like genres,taglines,overview can help us in finding similar movies so we select these features. Movies can have similar budget, revenue etc but be completely different from each other so we drop we these features.



final_df['tagline'] = final_df['tagline'].fillna('NA')
final_df.dropna(inplace=True)




final_df['overview'].astype(str)
final_df['movie_tags'] = final_df['overview'].apply(lambda x: x.lower() and x.split() )



#Defining a function to convert string values (content) to list (tags)
import ast
def contenttotags(x):
    lst = []
    for i in ast.literal_eval(x):
        lst.append(i['name'].replace(" ","").lower())
    return lst

# Apply Function 
final_df['movie_tags'] = final_df['movie_tags'] + final_df['genres'].apply(contenttotags)
final_df['movie_tags'] = final_df['movie_tags'] + final_df['keywords'].apply(contenttotags)

# Defining Extract the first 5 names from cast
def extractcast(X):
    lst = []
    counter = 0
    for i in ast.literal_eval(X):
        if counter < 5:
            lst.append(i['name'].replace(" ","").lower())
            counter += 1

    return lst

#Adding tags from cast column to the movie tags column
final_df['movie_tags'] = final_df['movie_tags'] + final_df['cast'].apply(extractcast)


# Defining function to extract Director name/
def director(x):
    lst = []
    for x in ast.literal_eval(x):
        if x['job'] == "Director":
            lst.append(x['name'].replace(" ","").lower())

    return lst

#Adding director tags to the movie tags column. 
final_df['movie_tags'] = final_df['movie_tags'] + final_df['crew'].apply(director)

# Convert movie title in list and concatenate with tags
final_df['movie_tags'] = final_df['movie_tags'] + final_df['title'].apply(lambda x: x.lower() and x.split() )

# Converting taglines (content) to a list of tags. 
final_df['tagline'] = final_df['tagline'].astype(str)
final_df['tagline'].apply(lambda x: x.lower() and x.split() )

#Adding tags from the tagline column to movie tags. 
final_df['movie_tags'] = final_df['movie_tags'] + final_df['tagline'].apply(lambda x: x.lower() and x.split() )


# Create new DataFrame with required columns only
final = final_df[['id', 'title', 'movie_tags']]

# Transform list of tags in the string and in lowercase
final['movie_tags'] = final['movie_tags'].apply(lambda x: " ".join(x).lower())
final['movie_tags'] = final['movie_tags'].str.lower()

import nltk
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

# Apply steming on the movie tags columns 
final['movie_tags'] = final['movie_tags'].apply(porter.stem)

# Remove Stop-words
from sklearn.feature_extraction.text import CountVectorizer
countVec = CountVectorizer(max_features=10000, stop_words='english')

# Convert Tags into vectors. 
Vectors = countVec.fit_transform(final['movie_tags']).toarray()




# Import cosine_similarity to Calulate Distance between the Vectors
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the Distance of all Movies with eachothers
similarity = cosine_similarity(Vectors)


# Recommender Function to Return Movie Names Only
def recommend_mmovies(Movie):
    movie_index = final[final['title'] == Movie].index[0]
    distances = sorted(list(enumerate(similarity[movie_index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    for i in distances[1:6]:
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names       

st.title('Movie Recommedation System')
st.subheader('By Krishna Bakshi - www.linkedin.com/in/krishnabakshi')
st.subheader('Code - ')
movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Select or Type a movie.",
    movie_list)

if st.button('Click to see recommendations!'):
    movienames = recommend_mmovies(selected_movie)
    for i in movienames:
        st.write(i)    
