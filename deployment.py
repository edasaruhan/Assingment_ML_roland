import pandas as pd
from flask import Flask, request, render_template, session
from pickle import load
import numpy as np
from logging.config import dictConfig
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] [%(levelname)s | %(module)s] %(message)s",
                "datefmt": "%B %d, %Y %H:%M:%S %Z",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "webSiteVisiters.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'axcipherkeyclassic'

# Load the model
model = load(open('model_version.pkl', 'rb'))
# Load the scaler
scaler = load(open('scaler_version1.pkl', 'rb'))


@app.route("/")
def home():
    session["ctx"] = {"request_id": str(uuid.uuid4())}

    app.logger.info("A user visited the home page >>> %s", session["ctx"])
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def main():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    prediction_str = "unstable"
    if prediction[0] == 1:
        prediction_str = "stable"

    return render_template("index.html", prediction_text="Stability of the Smart"
                                                         "grid: {}".format(prediction_str))


if __name__ == '__main__':
    app.run(debug=True, port=5000)
