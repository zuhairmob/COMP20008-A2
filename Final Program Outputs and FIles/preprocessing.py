import pandas as pd
import numpy as np
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.impute import KNNImputer
from numpy import mean
from numpy import std

def main():
    # Read in original csv files
    credits = pd.read_csv("credits.csv")
    titles = pd.read_csv("titles.csv", na_values='?')
    
    # Group actors, characters and directors by show/movie and return them in an easily accessible
    # form
    preprocessed_credits_df = preprocess_credits(credits)

    n_titles = len(titles.axes[0])
    
    # Create bag of words for the titles and description    
    names_df = titles['title']
    desc_df = titles['description']
    names_bow_df = bag_of_wordsify_df(names_df, n_titles, 'title')
    desc_bow_df = bag_of_wordsify_df(desc_df, n_titles, 'description')
    
    # Discretize titles by release year into 'OLD', 'MEDIUM' and 'NEW' 
    discretized_ages_df = discretized_release(titles)
    
    # Encode age certification into a scale between 1-5 (1 being all audiences, 5 being 18+)
    encoded_ratings_df = encode_age_cert(titles)

    # Impute missing information
    imputed_popularity_votes = impute_votes_popularity(titles)
    imputed_scores = impute_scores(titles)
    
    
    # Create a new dataframe, and append previously created dataframes which have been totally 
    # preprocessed. The resulting dataframe should be completely preprocessed, ready for analysis
    preprocessed_titles_df = pd.DataFrame()
    preprocessed_titles_df["id"] = titles["id"]
    preprocessed_titles_df["title"] = names_bow_df
    preprocessed_titles_df["type"] = titles["type"]
    preprocessed_titles_df["description"] = desc_bow_df
    preprocessed_titles_df["release_year"] = discretized_ages_df
    
    imputed_age_df = impute_age(titles, preprocessed_titles_df, encoded_ratings_df)
    
    preprocessed_titles_df["age_certification"] = imputed_age_df
    preprocessed_titles_df["runtime"] = titles["runtime"]
    preprocessed_titles_df["genres"] = titles["genres"]
    preprocessed_titles_df["production_countries"] = titles["production_countries"]
    preprocessed_titles_df["seasons"] = titles["seasons"]
    preprocessed_titles_df["imdb_id"] = titles["imdb_id"]
    
    preprocessed_titles_df["imdb_score"] =  imputed_scores["imdb_score"]
    preprocessed_titles_df["imdb_votes"] = imputed_popularity_votes["imdb_votes"]
    preprocessed_titles_df["tmdb_popularity"] = imputed_popularity_votes["tmdb_popularity"]
    preprocessed_titles_df["tmdb_score"] =  imputed_scores["tmdb_score"]
    
    # Merge the preprocessed titles dataframe with the preprocessed credits datafram by title id
    preprocessed_all = preprocessed_titles_df.merge(preprocessed_credits_df, on='id', how= 'left')
    
    # Filter the titles into movie and show 
    movies_df = create_movie_df(preprocessed_all)
    shows_df = create_show_df(preprocessed_all)
    
    # Export these new files into a new csv file to be analysed 
    movies_df.to_csv("movies.csv", index = False)
    shows_df.to_csv("shows.csv", index= False)
    return

def create_movie_df(titles):
    titles_type = titles['type']
    movies = []
    # Iterate through the entire csv, and if the title is a movie, append it to a new list
    for i in range(0,len(titles_type)):
        if (titles_type.iloc[i] == "MOVIE"):
            movie = titles.iloc[i].tolist()
            movies.append(movie)
    
    # Convert the movies list into a dataframe and return
    movies_df = pd.DataFrame(movies, columns = ['id','title','type','description','release_year',
                                                'age_certification','runtime','genres',
                                                'production_countries', 'seasons','imdb_id', 
                                                'imdb_score','imdb_votes','tmdb_popularity',
                                                'tmdb_score', 'actors', 'characters', 'director'])       
    return movies_df

def create_show_df(titles):
    titles_type = titles['type']
    shows = []
    # Iterate through the entire csv, and if the title is a show, append it to a new list
    for i in range(0,len(titles_type)):
        if (titles_type.iloc[i] == "SHOW"):
            show = titles.iloc[i].tolist()
            shows.append(show)
            
    # Convert the shows list into a dataframe and return
    shows_df = pd.DataFrame(shows, columns = ['id','title','type','description','release_year',
                                              'age_certification','runtime','genres',
                                              'production_countries', 'seasons','imdb_id', 
                                              'imdb_score','imdb_votes','tmdb_popularity',
                                              'tmdb_score', 'actors', 'characters', 'director'])       
    return shows_df

def discretized_release(titles):
    release_years = titles['release_year']
    discretized_age = []
    
    # If the title was released before 2010, group into 'OLD' category. Between 2010-2020 
    # (exclusive), group into 'MEDIUM' and if it was release after 2020 (inclusive), group into 
    # 'NEW'
    for i in range(0, len(release_years)):
        if (release_years.iloc[i] <= 2010):
            discretized_age.append("OLD")
        elif (2010 < release_years.iloc[i] <= 2019):
            discretized_age.append("MEDIUM")
        elif (2019 < release_years.iloc[i]):
            discretized_age.append("NEW")
            
    # Convert to dataframe and return
    discretized_age_df = pd.DataFrame(discretized_age, columns = ["release_year"])
    return discretized_age_df

def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    
    # Convert string into lowercase, if entry is NaN or an integer, convert it to string  
    if(type(text) == str):
        lowered = text.lower()
    else:
        lowered = str(text)
       
    # Remove punctuation and tokenize word     
    punct = r"([^a-z \d])"
    remove_punct = r""
    words = re.sub(punct, remove_punct, lowered)
    tokenized = nltk.word_tokenize(words)
    
    # Remove stopwords from the word list and lemmatie for more accurate analysis 
    stop = stopwords.words('english')
    no_stopwords = []
    for word in tokenized:
        if (word not in stop) and (word not in no_stopwords):
            lemmatized_word = lemmatizer.lemmatize(word)
            no_stopwords.append(lemmatized_word)
    return no_stopwords

def bag_of_wordsify_df(texts_df, n_rows, column_name):
    preprocessed_texts = []
    
    # Create a list of preprocessed texts and return list
    for i in range(0,n_rows):
        text = texts_df.iloc[i]
        preprocessed_text = preprocess(text)
        preprocessed_texts.append(preprocessed_text)
    return preprocessed_texts
    
def encode_age_cert(titles):
    age_rating = titles['age_certification']
    encoded_ratings = []
    
    # Iterate through the titles. Convert into: 1 if rating is 'TV-Y', 'TV-G' or 'G'; 2 if rating is
    # 'PG' or 'TV-PG'; 3 if rating is 'PG-13'; 4 if rating is 'TV-14' or 'R' and 5 if rating is 
    # 'TV-MA' or 'NC-17'
    for rating in age_rating:
        if(type(rating) == str):
            if(("TV-Y" in rating) or (rating == "TV-G") or (rating == "G")):
                encoded_ratings.append(1)
            elif((rating == "PG") or (rating == "TV-PG")):
                encoded_ratings.append(2)
            elif((rating == "PG-13")):
                encoded_ratings.append(3)
            elif((rating == "TV-14") or (rating == "R")):
                encoded_ratings.append(4)
            elif((rating == "TV-MA") or (rating == "NC-17")):
                encoded_ratings.append(5)
        # If there is missing information, append a 'NaN' 
        else:
            encoded_ratings.append(rating)
            
    # Convert to dataframe and return
    encoded_ratings_df = pd.DataFrame(encoded_ratings, columns = ['age_certification'])
    return encoded_ratings_df

def impute_age(titles, preprocessed_titles, encoded_ratings_df):
    imputer = KNNImputer(n_neighbors= 2)
    
    # Create an array of all unique genres and words found in title descriptions 
    genres_array = array_of_words(titles, "genres")
    desc_array = array_of_words(preprocessed_titles, "description")
    
    # Use K-Nearest Neighbour algorithm to impute age ratings (1-5) based on the genres and 
    # description words 
    IMPUTE_ME = pd.concat([encoded_ratings_df, genres_array, desc_array], 
                          axis = 'columns')
    imputed = imputer.fit_transform(IMPUTE_ME) 
    
    imputed_age = imputed [:,0]
    # Convert to dataframe and return the newly imputed age dataframe 
    imputed_age_df  = pd.DataFrame(imputed_age)
    return imputed_age_df

def impute_votes_popularity(titles):
    # Create a copy of the DataFrame to avoid modifying the original
    impute_df = titles.copy()

    # Select the columns for KNN imputation (tmdb_popularity, imdb_votes)
    selected_columns = ["tmdb_popularity", "imdb_votes"]
    imputer = KNNImputer(n_neighbors=2)

    # Perform imputation on the selected columns for both directions
    imputed_data_both_ways = imputer.fit_transform(impute_df[selected_columns])

    # Update the DataFrame with imputed values (both directions)
    impute_df["imdb_votes"] = imputed_data_both_ways[:, 1]
    impute_df["tmdb_popularity"] = imputed_data_both_ways[:, 0]

    # Return both imputed columns
    return impute_df[["tmdb_popularity", "imdb_votes"]]

def impute_scores(titles):
    # Create a copy of the DataFrame to avoid modifying the original
    impute_df = titles.copy()

    # Select the columns for KNN imputation (tmdb_score, imdb_score)
    selected_columns = ["tmdb_score", "imdb_score"]

    imputer = KNNImputer(n_neighbors=2)

    # Perform imputation on the selected columns for both directions
    imputed_data_both_ways = imputer.fit_transform(impute_df[selected_columns])

    # Update the DataFrame with imputed values (both directions)
    impute_df["imdb_score"] = imputed_data_both_ways[:, 1]
    impute_df["tmdb_score"] = imputed_data_both_ways[:, 0]

    # Return both imputed columns
    return impute_df[["tmdb_score", "imdb_score"]]

def array_of_words(titles, column_name):
    # Create an array of words of every unique string in the column 
    titles_lst = titles[column_name].apply(bracket_strip)
    split = titles_lst.str.join("").str.get_dummies(sep=' ')
    return split

def bracket_strip(str_of_lst):
    # Debugging, used in 'array_of_words' function to help remove additional punctuation in strings
    new = str(str_of_lst).strip("[]").split(',')
    return new

def preprocess_credits(credits):
    id_actor_dict = {}
    id_character_dict = {}
    id_director_dict = {}
    n_entries = len(credits.axes[0])
    
    # Iterate through credits csv, if person is an actor, append their name and character to the 
    # actor and character dictionary. If the person is a director, append their name to the 
    # director dictionary. Group by title id
    for person_index in range(0, n_entries):
        id = credits.iloc[person_index]["id"]
        name = name_preprocess(credits.iloc[person_index]["name"])
        role = credits.iloc[person_index]["role"]
        if (id not in id_actor_dict and (role == "ACTOR")):
            id_actor_dict[id] = name.split(maxsplit = 0)
            character = name_preprocess(credits.iloc[person_index]["character"])
            if (id not in id_character_dict):
                id_character_dict[id] = character.split(maxsplit = 0)
            else:
                id_character_dict[id].append(character)
               
        elif (id in id_actor_dict and (role == "ACTOR")):
            id_actor_dict[id].append(name)
            character = name_preprocess(credits.iloc[person_index]["character"])
            if (id not in id_character_dict):
                id_character_dict[id] = character.split(maxsplit = 0)
            else:
                id_character_dict[id].append(character)
                
                
        elif (id not in id_director_dict and (role == "DIRECTOR")):
            id_director_dict[id] = name.split(maxsplit = 0)
        elif(id in id_character_dict and (role == "DIRECTOR")):
            id_director_dict[id].append(name)
    
    # Convert dictionaries into one dataframe, by title id and return   
    for id in id_actor_dict:
        id_actor_dict[id] = str(id_actor_dict[id])
    for id in id_character_dict:
        id_character_dict[id] = str(id_character_dict[id])
    for id in id_director_dict:
        id_director_dict[id] = str(id_director_dict[id])
    
    id_actor_list = list(id_actor_dict.items())
    id_character_list = list(id_character_dict.items())
    id_director_list = list(id_director_dict.items())
    
    id_actor_df = pd.DataFrame(id_actor_list, columns = ["id", "actors"])
    id_character_df = pd.DataFrame(id_character_list, columns = ["id", "characters"])
    id_director_df = pd.DataFrame(id_director_list, columns = ["id", "director"])
    
    preprocessed_credits = id_actor_df.merge(id_character_df, on='id', how='left')
    preprocessed_credits = preprocessed_credits.merge(id_director_df, on='id', how= 'left')
    
    preprocessed_credits.to_csv("preprocessed_credits.csv", index= False)
    return preprocessed_credits

def name_preprocess(name):
    # Preprocess text, but for names (no lemmatizing etc), if there is no entry append 'NaN'
    if(type(name) == str):
        lowered = name.lower()
    else:
        lowered = str(name)
    punct = r"([^a-z \d])"
    remove_punct = r""
    words = re.sub(punct, remove_punct, lowered)
    return words
if __name__ == "__main__": 
    main()
