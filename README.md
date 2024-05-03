# AI-Project-Final : Fashion Recommendation System

Authors: Chris Haleas, Andrew Eby, Sanmeet Singh, and Rehaan Rafi

## Project Description
Many people struggle with deciding what to wear or what clothing to buy due to a lack of available fashion inspiration. So, we created an AI system that uses machine learning models trained on a large dataset of fashion items from Kaggle. This system interacts with users to understand their clothing preferences and provides personalized recommendations.

## Dependencies

- Pandas: [Download](https://pandas.pydata.org/docs/getting_started/install.html)
    - Used for data manipulation, particularly for reading and processing CSV files.
- Sklearn: [Download](https://scikit-learn.org/stable/install.html)
    - Provides machine learning algorithms and tools for data preprocessing, modeling, and evaluation.   
- CV2: [Download](https://pypi.org/project/opencv-python/)
    - Used for displaying images.

## Dataset
Our dataset consists of clothing attributes matched with a display name of the clothing. Each clothing item is also associated with an ID that corresponds to an image of the clothing. The text portion of the dataset is imported as "styles.csv" and is included in our project repository.

Due to GitHub's restrictions on large files, the image portion of the dataset is provided as a compressed file on Kaggle. **You can download the compressed image dataset from the following link**:

https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small?select=styles.csv

Once downloaded, extract the contents of the zip file. Import the "images" folder from the extracted files into your project folder. This will allow the system to display images that correspond to the recommendation names provided.

Please note that you need to have the "images" folder in your project directory for the image display functionality to work correctly.


...
**Note:** If you would like to use a dataset filtered for western apparel, use `filtered_western_apparel.csv`. Simply change any mention of `pd.read_csv('styles.csv'...)` to `pd.read_csv('filtered_western_apparel.csv'...)` in the `options.py`, `display.py`, `knn.py`, `forest.py`, and `dectree.py` files.
...

## How to use the Recommendation System 

1. **Run the Program:** Execute the `main.py` script in your terminal or preferred Python environment.

2. **Choose a Model:** You will be presented with three model options: K Nearest Neighbors, Random Forest, and Decision Tree. Enter the number corresponding to the model you would like to use.

3. **Provide Clothing Preferences:**
   - **Gender:** Enter your gender preference (e.g., Men, Women, Girls, Boys, Unisex).
   - **Clothing Type:** Select the type of clothing you are looking for (e.g., Topwear, Bottomwear, Innerwear, etc.).
   - **Specific Clothing:** Choose a specific type of clothing from the available options (e.g., Shirts, Tshirts, Tops, etc.).
   - **Color:** Specify the color of the clothing item you desire from the provided options.
   - **Season:** Indicate the season in which you intend to wear the clothing (e.g., Summer, Fall, Winter, Spring).
   - **Usage Context:** Select the context in which you plan to use the clothing (e.g., Ethnic, Casual, Formal).

4. **Receive Recommendation:** Based on your preferences, the system will recommend a clothing item. The output will display the recommended clothing item, the accuracy score based on correct attributes, and the cosine similarity score between your input and the recommended item.

5. **View Image:** The system will attempt to display an image corresponding to the recommended clothing item. Note that this functionality depends on having the correct image files in the `images` folder of your project directory.

6. **Repeat or Exit:** You can run the program multiple times with different preferences to explore different clothing recommendations. To exit the program, simply close the terminal or Python environment. 



## Description of Code
- **main.py**: The main menu of the project.
    - This script allows the user to choose between three different machine learning models for clothing recommendation: K Nearest Neighbors, Random Forest, and Decision Tree. The user is prompted to enter the number corresponding to the desired model, and the script then imports and executes the chosen model from separate files (`knn.py`, `forest.py`, `dectree.py`).

- **knn.py**: The file for our KNN model.
    - **Data Handling**: Reads the clothing data from the 'styles.csv' file using `pd.read_csv` and handles missing values in the 'productDisplayName' column using `SimpleImputer(strategy='most_frequent')`.
    - **Data Preparation**: Prepares the data for modeling by separating the features ('gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'usage') and target variable ('productDisplayName').
    - **Data Encoding**: Encodes categorical features using `OneHotEncoder` within a `ColumnTransformer`.
    - **Model Creation**: Creates a K Nearest Neighbors (KNN) classifier model using `KNeighborsClassifier(n_neighbors=5, metric='cosine')` within a `Pipeline` with the preprocessor.
    - **User Interaction**: Prompts the user for clothing preferences, including gender, category, type, color, season, and usage context, and creates a DataFrame with this input. This user interaction is called from the `filter_clothing_data()` function of `options.py`. 
    - **Prediction**: Uses the KNN model to predict a matching clothing item based on the user's input.
    - **Accuracy Calculation**: Calculates an accuracy score based on the match between the user's preferences and the predicted item, computing the percentage of matching attributes.
    - **Cosine Similarity**: Computes the cosine similarity between the user's preferences and the predicted item, indicating the similarity between them.
    - **Image Display**: Calls `display_image(predicted_product)` from `display.py` to show an image of the recommended clothing item.

- **forest.py**: The file for our Random Forest Classifier model,
    - Performs identical operations to the `knn.py` file but uses a Random Forest Classifier instead.

- **dectree.py**: The file for our Decision Tree Classifier model.
    - Performs identical operations to the `knn.py` file but uses a Decision Tree Classifier instead.

- **options.py**: The file that `knn.py`, `forest.py`, and `dectree.py` call to get user input. 
    - Reads the 'styles.csv' file and filters the data to include only apparel items.
    - Prompts the user to select preferences for gender, clothing type, color, season, and usage context.
    - Filters the data based on the user's selections and returns a dictionary representing the user's input.
    - The returned dictionary is passed to the models for further processing.

- **display.py**: The file that `knn.py`, `forest.py`, and `dectree.py` call to display recommendation images. 
    - Reads the 'styles.csv' file to find the image corresponding to the recommended product display name and displays it using OpenCV (`cv2`).
## Credit 
All credit for dataset goes to Param Aggarwal. Link to original dataset: 
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
