# -*- coding: utf-8 -*-
import json
import os
import pathlib
from typing import Tuple

import flasgger
import flask
import numpy as np
import pandas as pd
from sklearn import base, ensemble, preprocessing
from sklearn.externals import joblib

from data import make_dataset  # pylint: disable=import-error

app = flask.Flask(__name__)
app.config['SWAGGER'] = {
    'termsOfService':
    None,
    'specs': [{
        'version': '0.0.1',
        'title': 'A Sample Regressor Project',
        'description': 'An Apartments Price Regressor',
        'endpoint': 'v0_spec',
        'route': '/v0/spec'
    }]
}
swag = flasgger.Swagger(app)


def load_models() -> Tuple[base.RegressorMixin, preprocessing.StandardScaler]:
    project_dir = pathlib.Path(__file__).resolve().parents[1]
    model_file_path = os.path.join(project_dir, 'models', 'rf_model.joblib')
    scaler_file_path = os.path.join(project_dir, 'models', 'rf_scaler.joblib')
    with open(model_file_path, 'rb') as m, open(scaler_file_path, 'rb') as s:
        clf_loaded = joblib.load(m)
        scaler_loaded = joblib.load(s)
    return clf_loaded, scaler_loaded


model, scaler = load_models()


# POST /api/predict
@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint for retrieving predictions from the regressor.

    This endpoint accepts data in JSON format in the from of a dataframe in 
    split orientation and returns a map of predictions keyed by index.

    ---
    tags:
      - predictions
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: dataframe
        description: The dataframe to process
        required: true
    responses:
        200:
            description: prediction OK
    """
    request_data = flask.request.get_json()

    df = pd.read_json(json.dumps(request_data), orient='split')
    df_pro = make_dataset.process_data(df)

    values_std = scaler.transform(df_pro.values)

    predictions = model.predict(values_std)

    indexes = df_pro.index

    # Zip the indexes with the predictions so they can be identified
    response = flask.Response(
        json.dumps(dict(zip(indexes, predictions))),
        status=200,
        mimetype='application/json')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
