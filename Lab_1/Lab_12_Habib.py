# Lab12 : Classification of flours iris using scikit-learn
# Rèalisé par : Habib Tanou EMSI 2023 - 2024

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import streamlit as st
import pandas as pd

# Step 1: DataSet
iris = datasets.load_iris()
print(iris.data)
print(iris.target)
print(iris.target_names)
print(iris.feature_names)
print(iris.data.shape)


def user_input():
    sepal_length = st.sidebar.slider('sepal length', 0.1, 9.9, 5.0)
    sepal_width = st.sidebar.slider('sepal width', 0.1, 9.9, 5.0)
    petal_length = st.sidebar.slider('petal length', 0.1, 9.9, 5.0)
    petal_width = st.sidebar.slider('petal width', 0.1, 9.9, 5.0)
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    flower_features = pd.DataFrame(data, index=[0])
    return flower_features


st.sidebar.header('iris features')
df = user_input()
st.write(df)

# Step 2: Model
algo_dict = {
    'RandomForestClassifier': RandomForestClassifier(),
    'ExtraTreesClassifier': ExtraTreesClassifier()
}
selected_model = st.sidebar.selectbox('select your learning algorithm', list(algo_dict.keys()))
model = algo_dict[selected_model]

# Step 3: Train
model.fit(iris.data, iris.target)

# Step 4: Test
prediction = model.predict([[0.9, 1.0, 1.1, 1.8]])
print(prediction)
print(iris.target_names[prediction])

# Web Deployment of the Model: streamlit run filename.py
st.header('iris flower classification')
st.image('img/iris.png')

st.write('selected algorithm is: ', selected_model)
prediction = model.predict(df)
st.write("prediction", prediction)
st.write("Name of the predicted classe:", iris.target_names[prediction])
st.image('img/' + iris.target_names[prediction][0] + '.png')
