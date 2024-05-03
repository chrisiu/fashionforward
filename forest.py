import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from options import filter_clothing_data
from sklearn.metrics.pairwise import cosine_similarity
from display import display_image

def forest():
    data = pd.read_csv('styles.csv', on_bad_lines='skip')
    imputer = SimpleImputer(strategy='most_frequent')
    data['productDisplayName'] = imputer.fit_transform(data[['productDisplayName']])

    X = data[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']]
    y = data['productDisplayName']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=95)

    categorical_features = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('forest', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=99))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)


    data1 = data
    data = filter_clothing_data()

    if data is not None:
        gender = data['gender']
        masterCategory = data['masterCategory']
        subCategory = data['subCategory']
        articleType = data['articleType']
        baseColour = data['baseColour']
        season = data['season']
        usage = data['usage']

        user_input = pd.DataFrame({
            'gender': [gender],
            'masterCategory': [masterCategory],
            'subCategory': [subCategory],
            'articleType': [articleType],
            'baseColour': [baseColour],
            'season': [season],
            'usage': [usage]
        })
        print(user_input)
    else:
        print("No similar item found.")

    prediction = pipeline.predict(user_input)

    print("")
    print("")
    print("Random Forest Classifier Output")
    print("___________________________________________")
    print(f"Recommended Clothing: {prediction[0]}")

    print("")
    print("")
    print("Random Forest Classifier Accuracy")
    print("___________________________________________")


    predictedName = prediction[0]
    predictedRow = data1[data1['productDisplayName'] == predictedName].iloc[0]

    predicted_df = pd.DataFrame({
        'gender': [predictedRow['gender']],
        'masterCategory': [predictedRow['masterCategory']],
        'subCategory': [predictedRow['subCategory']],
        'articleType': [predictedRow['articleType']],
        'baseColour': [predictedRow['baseColour']],
        'season': [predictedRow['season']],
        'usage': [predictedRow['usage']]
    })

    user_input = preprocessor.transform(user_input)
    predicted_df = preprocessor.transform(predicted_df)

    score = cosine_similarity(user_input, predicted_df)
    print("Cosine similarity: ", score[0][0] * 100, "%")



    display_image(prediction[0])
