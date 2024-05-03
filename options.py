import pandas as pd

def filter_clothing_data():
    try:
        df = pd.read_csv('styles.csv', on_bad_lines='skip')

        df = df[df['masterCategory'] == 'Apparel']


        gender_options = df['gender'].unique()

        print("\nGender Options:")
        print(gender_options)

        while True:
            selected_gender = input("\nWhat gender option are you looking for? ").capitalize()
            if selected_gender in map(str.capitalize, gender_options):
                break
            else:
                print("Invalid option. Please select from the given options.")

        sub_categories = df['subCategory'].unique()

        print("\nClothing Options:")
        print(sub_categories)

        while True:
            selected_sub_category = input("\nWhat type of clothing are you looking for? ").capitalize()
            if selected_sub_category in map(str.capitalize, sub_categories):
                break
            else:
                print("Invalid option. Please select from the given options.")

        df_filtered = df[df['subCategory'].str.lower() == selected_sub_category.lower()]

        article_types = df_filtered['articleType'].unique()

        print("\nTypes of " + selected_sub_category + " clothing:")
        print(article_types)

        while True:
            selected_article_type = input("\nWhat type of " + selected_sub_category + " clothing are you looking for? ").capitalize()
            if selected_article_type in map(str.capitalize, article_types):
                break
            else:
                print("Invalid option. Please select from the given options.")

        df_filtered = df_filtered[df_filtered['articleType'].str.lower() == selected_article_type.lower()]

        base_colors = df_filtered['baseColour'].unique()

        print("\nColor Options:")
        print(base_colors)

        while True:
            selected_base_color = input("\nWhat color " + selected_article_type + " are you looking for? ").capitalize()
            if selected_base_color in map(str.capitalize, base_colors):
                break
            else:
                print("Invalid option. Please select from the given options.")

        df_filtered = df_filtered[df_filtered['baseColour'].str.lower() == selected_base_color.lower()]

        seasons = df_filtered['season'].unique()

        print("\nSeason Options:")
        print(seasons)

        while True:
            selected_season = input("What season would you wear your " + selected_article_type + " in? ").capitalize()
            if selected_season in map(str.capitalize, seasons):
                break
            else:
                print("Invalid option. Please select from the given options.")

        df_filtered = df_filtered[df_filtered['season'].str.lower() == selected_season.lower()]

        usages = df_filtered['usage'].unique()

        print("\nUsage Options: ")
        print(usages)

        while True:
            selected_usage = input("In what context are you looking to use your " + selected_article_type + "? ").capitalize()
            if selected_usage in map(str.capitalize, usages):
                break
            else:
                print("Invalid option. Please select from the given options.")

        user_input = {
            'gender': selected_gender,
            'masterCategory': "Apparel",
            'subCategory': selected_sub_category,
            'articleType': selected_article_type,
            'baseColour': selected_base_color,
            'season': selected_season,
            'usage': selected_usage
        }

        return user_input

    except pd.errors.ParserError as e:
        print("Error parsing CSV file. Please check the file format.")
        print(e)

    except FileNotFoundError as e:
        print("File not found. Please check the file path.")
        print(e)

