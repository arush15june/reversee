# Reverse Image

CNN to recommend similar T-shirt's from the database

- model/scraper/scraper.py to scrape the images
- model/scraper/resize.py to crop them to 130x150 images

Use the flask app (app/application.py) to access the trained model (model.h5) using the REST API (/api/match)

`curl -X PUT -F 'image=@file.jpg' http://localhost:5000/api/match` 