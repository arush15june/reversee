# ReverSEE

CNN to recommend similar T-shirt's from the database

# Dataset
Cannot put the URL for the dataset publicly, but you can email me if you are interested. 

- `model/scraper/scraper.py` to scrape the images
- `model/scraper/resize.py` to crop them to 130x150 images

# Training the model

- It uses PyTorch with CUDA to train. If you don't have a GPU, please replace all the `.cuda()` variables.

- use `model/train.py` to train the model
    - parameters: 
        - `--images_dir` : Path with resized images
        - `--batch_size` : Training Batch Size
        - `--epochs` : No Of Epochs To Train For
        - `--savefile` : Save model with filename
# Using the model
- You can use the `QueryTest.ipynb` notebook for testing against images in the folder
- You can use the `test.ipynb` for testing against images in the dataset
- Use the Flask app (`app/application.py`) to access the trained model (`model.h5`) using the REST API (`/api/match`)

    ```curl -X PUT -F 'image=@file.jpg' http://localhost:5000/api/match``` 

Using the REST API is not practical at the moment, because with a dataset of 10000 images it takes 100 seconds to find the top 10 matches.
# Attributions 
- https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch

# TODO
- Compare with a regular CNN.
- Frontend for API.