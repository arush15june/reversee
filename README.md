# ReverSEE

CNN to recommend similar T-shirt's from the database

# Dataset
Cannot put the URL for the dataset publicly, but you can email me if you are interested. 

- `model/scraper/scraper.py` to scrape the images
- `model/scraper/resize.py` to crop them to 130x150 images

# Training the model

- It uses PyTorch with CUDA to train. If you don't have a GPU, replace all the `.cuda()` variables.

- There are two different models, A regular CNN and a Siamese Network Model.
- To use the Regular CNN
    - Use `model/scraper/gen_classes.py` to generate the labels for the model 
    - Use `model/train_cnn.py` to train the model. Customize the batch size to what your GPU can accomodate.
        - parameters
            - `--images_dir` : Path with resized images
            - `--batch_size` : Training Batch Size
            - `--epochs` : No Of Epochs To Train For
            - `--savefile` : Save model with filename
            - `--learnrate` : Learning Rate
            - `--labelfile` : CSV Containing the labels
- To use the Siamese Network
    - use `model/train.py` to train the model
        - parameters: 
            - `--images_dir` : Path with resized images
            - `--batch_size` : Training Batch Size
            - `--epochs` : No Of Epochs To Train For
            - `--learnrate` : Learning Rate
            - `--savefile` : Save model with filename


# Using the model
- You can use the `QueryTest.ipynb` notebook for testing against images in the folder
- You can use the `test.ipynb` for testing against images in the dataset
- Use the Flask app (`app/application.py`) to access the trained model (`model.h5`) using the REST API 
    - `/api/match` for the Siamese Network
        - ```curl -X PUT -F 'image=@file.jpg' http://localhost:5000/api/match``` 
    - `/api/matchcnn` for the Regular CNN
        - ```curl -X PUT -F 'image=@file.jpg' http://localhost:5000/api/matchcnn``` 

Using the REST API with the Siamese Network is not practical at the moment, because with a dataset of 10000 images it takes 100 seconds to find the top 10 matches.

# Frontend
- A frontend is present inside `/docs` to access the REST API (thanks @kautukkundan)
- Also hosted on `arush15june.github.io/reversee`

# Attributions 
- https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch

# TODO
- Siamese Network and CNN Comparison
