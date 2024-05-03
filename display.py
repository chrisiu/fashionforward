import cv2
import pandas as pd
import numpy as np

def display_image(predicted_product):

    data = pd.read_csv('styles.csv', on_bad_lines='skip')
    
    product_row = data.loc[data['productDisplayName'] == predicted_product]

    if not product_row.empty:
        image_id = product_row['id'].values[0]
        image_path = f"images/{image_id}.jpg"

        image = cv2.imread(image_path)
        if image is not None:
            cv2.imshow("Top Result", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"No image found for '{predicted_product}'")
    else:
        print(f"No matching product found for '{predicted_product}'")
