# COMP20008 Assignment 3 
preprocessing.py is a function which reads two csv files: "titles.csv" and "credits.csv".
In "titles.csv", the columns "title", "description", "release_year", "age_certification", "imdb_score", "imdb_votes", "tmdb_popularity" and "tmdb_score" are preprocessed and standardised, so that it can be analysed later. 
Rows in "credits.csv" is grouped by "id", and all of the actors, characters and the director(s) are collated together, with strings standardised for analysis. 
The program merges these two preprocessed csv files together, and then splits them according to "type", which determines whether the title is a "SHOW" or a "MOVIE", creating a separate csv for each, titled "shows.csv" and "movies.csv" 
respectively.
"age_certification" is encoded (number between 1-5), so that movies and shows will have equivalent age ratings (ie NC-17 and TV-MA as the highest rating for movies and shows)
For empty cells, in columns "age_certification", "imdb_score", "imdb_votes", "tmdb_popularity", "tmdb_score", K-nearest neighbours algorithm is used in order to impute the values. "age_certification" is imputed based off of unique
strings found in "description" and "genre". "imdb_votes" is imputed from "tmdb_popularity" and vice versa, and "imdb_score" is imputed from "tmdb_score" and vice versa. 
The columns "title" and "description" are text processed: strings are casefolded, stopwords are removed, tokenized into individual words and then lemmatized (for comparison reasons later). 
"release_year" is discretitized into categories (OLD, MEDIUM and NEW), based on the release year. Titles released before 2010 are classified as "OLD", 2011-2019 as "MEDIUM" and >2020 as "NEW". This was chosen so that each category
had approximately the same number of titles. 

HOW TO RUN:
(preprocessing function)
''' 
python preprocessing.py
'''

supervised_models_shows.py & python supervised_models_movies.py   are python programs that demonstrate how to build and evaluate linear regression models for predicting IMDb and TMDB scores of TV shows and Movies based on various numeric features. It utilizes the pandas, numpy, scikit-learn, and matplotlib libraries to perform data manipulation, model training, evaluation, and visualization. Both programs run essentially the same however, supervised_models_shows.py utilises the preprocessed shows dataset as well as introduces an additional feature 'seasons' to the model.


The dataset is initially split into two separate sets: one for IMDb scores and another for TMDB scores. Each set is further divided into training and testing datasets using train_test_split from scikit-learn. Two linear regression models are created, one for predicting IMDb scores and another for TMDB scores, using scikit-learn's LinearRegression. 

The models are trained on the training datasets. Predictions are made on the testing datasets. The program calculates Mean Squared Error (MSE) and R-squared (R2) scores to evaluate the model's performance for both IMDb and TMDB scores. Matplotlib is used to create visualizations of the regression models and feature importance plots for both IMDb and TMDB scores. The program prints the performance metrics (MSE and R2) for both IMDb and TMDB model

HOW TO RUN:
(supervised_models_shows.py & supervised_models_movies.py)
''' 
python supervised_models_shows.py / python supervised_models_movies.py 

unsupervised_model_movies.py & unsupervised_model_shows.py perform Principal Component Analysis (PCA) on the preprocessed movies and show datasets respectively. PCA is a dimensionality reduction technique that helps in understanding the underlying structure and patterns in data by transforming the original features into a new set of uncorrelated variables called principal components. This code utilizes the pandas, numpy, scikit-learn, and matplotlib libraries for data manipulation, standardization, PCA, and visualization.

Once data is loaded from respective datasets, A subset of features ('runtime,' 'seasons' (for shows only), 'imdb_votes,' 'tmdb_popularity,' 'age_certification') is selected for PCA. These selected features are standardized to have a mean of 0 and a standard deviation of 1 using StandardScaler from scikit-learn, ensuring all features are comparable by being of the same scale. PCA is then performed by utilising the PCA class from scikit-learn. It computes the principal components and their corresponding explained variance ratios.
Two Plots are then generated:
Scree Plot: This plot displays the variance ratio for each principal component, helping to determine how many components capture most of the data's variation.
Cumulative Explained Variance Plot: This plot shows the cumulative explained variance as the number of principal components increases.
Feature Loadings in Principal Components: A scatter plot is created to visualize the feature loadings in the first two principal components. It helps interpret the relationships between the original features and the principal components.

The code prints out useful information as a standard text output, including the explained variance ratios for each principal component, cumulative explained variance, and loadings of original features in the first two principal components.

HOW TO RUN:
(unsupervised_model_movies.py & unsupervised_model_shows.py)
''' 
python unsupervised_models_shows.py / python unsupervised_models_movies.py 