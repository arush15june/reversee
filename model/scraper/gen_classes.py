import os
import pandas as pd

IMAGES_DIR = 'images'

ALL_IMAGES = os.listdir(IMAGES_DIR)

classes = []
index = 0

for file_name in ALL_IMAGES:
    if 'jpg' not in file_name:
        continue

    classes.append({
        'file' : file_name,
        'class' : index
    })    

    index += 1

df = pd.DataFrame(classes)
print(df)
df.to_csv(IMAGES_DIR+'/classes.csv', index=False)
