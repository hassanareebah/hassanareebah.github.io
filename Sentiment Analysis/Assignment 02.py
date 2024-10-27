"""
This module performs sentiment analysis on reviews using multiple classification models.
It preprocesses the data, evaluates model performance, and identifies the best model 
using hyperparameter tuning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

data = pd.read_csv('reviews.csv', delimiter='\t')


def assign_sentiment(rating):
    """
    Assign a sentiment label based on the given rating.

    Parameters:
    rating (int): The rating value to classify.

    Returns:
    int: 0 for negative, 1 for neutral, and 2 for positive sentiment.
    """
    if rating in [1, 2]:
        return 0
    elif rating == 3:
        return 1
    elif rating in [4, 5]:
        return 2


data['Sentiment'] = data['RatingValue'].apply(assign_sentiment)

# Check the distribution of sentiments
sentiment_counts = data['Sentiment'].value_counts()
print(sentiment_counts)

# Create a balanced dataset based on the minimum class count
min_count = sentiment_counts.min()

balanced_data = pd.concat([
    data[data['Sentiment'] == 0].sample(min_count, random_state=1),
    data[data['Sentiment'] == 1],
    data[data['Sentiment'] == 2].sample(min_count, random_state=1)
])

balanced_data = balanced_data.sample(frac=1, random_state=1).reset_index(drop=True)
balanced_data['Sentiment'].value_counts()

# Split data into training & validation datasets and save to CSV files
train, valid = train_test_split(balanced_data, test_size=0.2, random_state=1)
train.to_csv('train.csv', index=False)
valid.to_csv('valid.csv', index=False)

# Load train and validation data
train_data = pd.read_csv('train.csv')
valid_data = pd.read_csv('valid.csv')


"""
Set up multiple classification models with pipelines and performs hyperparameter tuning 
using GridSearchCV. Evaluates model accuracy through cross-validation and identifies the best 
parameters for the Logistic Regression model.
"""

models = {
    'KNN': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

pipelines = {name: Pipeline([('vectorizer', CountVectorizer()), ('classifier', clf)]) for name, clf in models.items()}

for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, train_data['Review'], train_data['Sentiment'], scoring='accuracy', cv=5)
    print(f'{name} Accuracy: {scores.mean():.2f} ± {scores.std():.2f}')


"""
Identifies Logistic Regression as the best-performing model based on evaluation results.
"""

param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs']
}

logreg_pipeline = pipelines['Logistic Regression']
grid_search = GridSearchCV(logreg_pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=-1)

grid_search.fit(train_data['Review'], train_data['Sentiment'])

print("Best Parameters for Logistic Regression:", grid_search.best_params_)
print("Best Cross-validation Score for Logistic Regression:", grid_search.best_score_)

best_logreg_pipeline = grid_search.best_estimator_
valid_predictions = best_logreg_pipeline.predict(valid_data['Review'])

def evaluate(best_model, filename):
    """
    Evaluate the best model on the validation dataset and print performance metrics.

    Parameters:
    best_model: The model to evaluate.
    filename (str): Path to the validation dataset CSV file.
    """
    valid_data = pd.read_csv(filename)
    valid_predictions = best_model.predict(valid_data['Review'])  

    accuracy = accuracy_score(valid_data['Sentiment'], valid_predictions)
    f1_macro = f1_score(valid_data['Sentiment'], valid_predictions, average='macro')
    class_report = classification_report(valid_data['Sentiment'], valid_predictions, output_dict=True)

    print(f'Best Model (Logistic Regression) Evaluation:')
    print(f'Accuracy:  {accuracy:.2f}')
    print(f'Average F1 Score: {f1_macro:.2f}')
    print('Class-wise F1 Scores:')
    for sentiment, metrics in class_report.items():
        if sentiment in ['0', '1', '2']:  
            print(f'Sentiment {sentiment}: {metrics["f1-score"]:.2f}')

    conf_matrix = confusion_matrix(valid_data['Sentiment'], valid_predictions)
    conf_matrix = pd.DataFrame(conf_matrix, 
                                index=['Negative', 'Neutral', 'Positive'], 
                                columns=['Predicted Negative', 'Predicted Neutral', 'Predicted Positive'])
    print('Confusion Matrix:')
    print(conf_matrix)


# Evaluate the best model
evaluate(best_logreg_pipeline, 'valid.csv')
