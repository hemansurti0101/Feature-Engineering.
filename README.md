# Feature-Engineering.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Step 1: Read the books dataset and explore it
books = pd.read_csv("BX-Books.csv", sep=";", error_bad_lines=False, encoding="latin-1")
# Explore books dataset
print(books.head())

# Step 2: Clean up NaN values
books.dropna(inplace=True)

# Step 3: Read the data where ratings are given by users
ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", error_bad_lines=False, encoding="latin-1")
# Explore ratings dataset
print(ratings.head())

# Step 4: Take a quick look at the number of unique users and books
print("Number of unique users:", ratings['user_id'].nunique())
print("Number of unique books:", ratings['isbn'].nunique())

# Step 5: Convert ISBN variables to numeric numbers in the correct order
books['isbn'] = pd.factorize(books['isbn'])[0]

# Step 6: Convert the user_id variable to numeric numbers in the correct order
ratings['user_id'] = pd.factorize(ratings['user_id'])[0]

# Step 7: Convert both user_id and ISBN to the ordered list, i.e., from 0...n-1
ratings['isbn'] = pd.factorize(ratings['isbn'])[0]

# Step 8: Re-index the columns to build a matrix
ratings_matrix = ratings.pivot(index='user_id', columns='isbn', values='rating').fillna(0)

# Step 9: Split your data into two sets (training and testing)
train_data, test_data = train_test_split(ratings_matrix, test_size=0.2, random_state=42)

# Step 10: Make predictions based on user and item variables
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

# Step 11: Use RMSE to evaluate the predictions
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

# Example usage:
user_prediction = predict(train_data.values, user_similarity, type='user')
item_prediction = predict(train_data.values, item_similarity, type='item')

print('User-based CF RMSE:', rmse(user_prediction, test_data.values))
print('Item-based CF RMSE:', rmse(item_prediction, test_data.values))

