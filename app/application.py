"""

Barcode Reader API
    - github.com/arush15june
"""

import sys
from io import BytesIO

from flask import Flask, request, make_response, jsonify, Response
from flask_restful import Resource, Api, reqparse, abort
from flask_cors import CORS
from werkzeug.datastructures import FileStorage

from helpers import ModelPredictor, CNNModelPredictor

DEFAULT_MODEL_PATH = "model.h5"
DEFAULT_CNN_MODEL_PATH = "model_cnn_full.h5"
DEFAULT_IMAGES_DIR = "../model/scraper/images/"
DEFAULT_CLASSES_FILE = "classes_full.csv"


"""
    Flask Config
"""

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png']

app = Flask(__name__)
app.config.from_object(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

api = Api(app)

class FileStorageArgument(reqparse.Argument):
    """This argument class for flask-restful will be used in
    all cases where file uploads need to be handled."""
    
    def convert(self, value, op):
        if self.type is FileStorage:  # only in the case of files
            # this is done as self.type(value) makes the name attribute of the
            # FileStorage object same as argument name and value is a FileStorage
            # object itself anyways
            return value

        # called so that this argument class will also be useful in
        # cases when argument type is not a file.
        super(FileStorageArgument, self).convert(*args, **kwargs)


class Reverser(Resource):
    put_parser = reqparse.RequestParser(argument_class=FileStorageArgument)
    put_parser.add_argument('image', required=True, type=FileStorage, location='files')
    
    predictor = ModelPredictor(DEFAULT_MODEL_PATH, DEFAULT_IMAGES_DIR+"training")    

    @staticmethod
    def verify_extension(image):
        extension = image.filename.rsplit('.', 1)[1].lower()
        if '.' in image.filename and not extension in app.config['ALLOWED_EXTENSIONS']:
            return False
        else:
            return True

    @staticmethod
    def DFtoDict(df):
        return {'results': list(df.T.to_dict().values())}

    def put(self):
        args = self.put_parser.parse_args()
        image = args['image']

        if not self.verify_extension(image):
            abort(400, message='Unsupported File Extension')

        image_file = BytesIO()
        try:
            image.save(image_file)
        except:
            abort(400, message="Invalid Input")

        top10 = self.predictor.getMatches(image_file)
        data_dict = self.DFtoDict(top10)

        return make_response(jsonify(data_dict))
    
class ReverserCNN(Reverser):
    predictor = CNNModelPredictor(DEFAULT_CNN_MODEL_PATH, DEFAULT_CLASSES_FILE)

api.add_resource(Reverser, "/api/match")
api.add_resource(ReverserCNN, "/api/matchcnn")

if __name__ == '__main__':
    app.run(debug=True)