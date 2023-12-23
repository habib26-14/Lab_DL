# Lab12 : Classification of flours iris using scikit-learn
# Réalisé par : Habib Tanou EMSI 2023 - 2024

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import pandas as pd

# Amélioration 1: Afficher le nom et la figure; sa veut dire le nom et la photo.
iris = datasets.load_iris()

# Amélioration 2: Déployer l'app dans le web.
st.title('Iris Flower Classification App')
st.header('Iris flower classification')

# Step 1: DataSet
st.sidebar.header('Choose Features for Prediction')
iris_features = st.sidebar.selectbox('Select Iris Features', iris.feature_names)

def load_data():
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

df = load_data()
st.write(df)

def user_input():
    sepal_length = st.sidebar.slider('sepal length', df['sepal length (cm)'].min(), df['sepal length (cm)'].max(), df['sepal length (cm)'].mean())
    sepal_width = st.sidebar.slider('sepal width', df['sepal width (cm)'].min(), df['sepal width (cm)'].max(), df['sepal width (cm)'].mean())
    petal_length = st.sidebar.slider('petal length', df['petal length (cm)'].min(), df['petal length (cm)'].max(), df['petal length (cm)'].mean())
    petal_width = st.sidebar.slider('petal width', df['petal width (cm)'].min(), df['petal width (cm)'].max(), df['petal width (cm)'].mean())
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features

flower_input = user_input()

# Amélioration 3: Donner à l'utilisateur la possibilité de choisir parmi 10 algorithmes pour entrainer le modèle,
# et afficher le nom du modèle utilisé dans le training après avoir fait la classification.
st.sidebar.header('Select Model for Training')
algo_dict = {
    'RandomForestClassifier': RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'LogisticRegression': LogisticRegression(),
    'SVM': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'GaussianNB': GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(),
    'MLPClassifier': MLPClassifier(),
    'XGBClassifier': XGBClassifier()
}
selected_model = st.sidebar.selectbox('Select your learning algorithm', list(algo_dict.keys()))
model = algo_dict[selected_model]

# Amélioration 4: Expliquer chaque algorithme, son fonctionnement et ses avantages.
st.sidebar.header('Algorithm Explanation')
algorithm_info = {
    'RandomForestClassifier': 'Random Forest is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees.',
    'ExtraTreesClassifier': 'Extra Trees is an ensemble learning method similar to a random forest but more random. It constructs a multitude of decision trees at training time and outputs the class that is the mode of the classes (classification) of the individual trees.',
    'GradientBoostingClassifier': 'Gradient Boosting is an ensemble learning method that builds a series of weak learners and combines their predictions to improve accuracy.',
    'LogisticRegression': 'Logistic Regression is a linear model for binary classification that predicts the probability of occurrence of an event.',
    'SVM': 'Support Vector Machine is a powerful and versatile machine learning algorithm for classification and regression tasks.',
    'KNeighborsClassifier': 'K-Nearest Neighbors is a simple and effective algorithm for classification that classifies a data point based on the majority class of its k-nearest neighbors.',
    'GaussianNB': 'Naive Bayes is a probabilistic algorithm that makes classifications based on the likelihood of different events.',
    'DecisionTreeClassifier': 'Decision Tree is a tree-structured model that makes decisions based on rules learned from the training data.',
    'MLPClassifier': 'Multi-layer Perceptron is a type of artificial neural network that consists of multiple layers of nodes and can be used for both classification and regression tasks.',
    'XGBClassifier': 'XGBoost is an optimized distributed gradient boosting library designed for efficient and high-performance machine learning.',
}
selected_algorithm_info = algorithm_info.get(selected_model, 'No information available for the selected algorithm')
st.sidebar.text(selected_algorithm_info)

# Step 3: Train
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Step 4: Test
prediction = model.predict(X_test)
accuracy = accuracy_score(y_test, prediction)

# Amélioration 5: Comparer les métriques et la "accuracy" et expliquer comment on va la calculer.
st.sidebar.header('Model Evaluation Metrics')
st.sidebar.text(f'Accuracy: {accuracy:.2f}')
st.sidebar.text(classification_report(y_test, prediction))

# Web Deployment of the Model: streamlit run filename.py
st.sidebar.write('Selected algorithm for training:', selected_model)
st.sidebar.write('Model Training Accuracy:', accuracy)

st.header('Iris flower classification')
st.write('Selected algorithm for classification:', selected_model)
prediction = model.predict(flower_input)
st.write("Prediction:", prediction)
st.write("Name of the predicted class:", iris.target_names[prediction][0])
st.image(f'img/{iris.target_names[prediction][0]}.png')
