import numpy as np
import scipy.interpolate
from sklearn.linear_model import LogisticRegression
import resistive_pulse as rp
import csv
from sklearn.externals import joblib
import copy

class RPPredictor:

    num_features = 100
    num_points = 100

    def __init__(self):

        self._features = []

        self._training_events = []
        self._training_features = []
        self._training_labels = []

        self._test_events = []
        self._test_features = []

        self._events = []
        self._features = []



    def load_training_files(self, file_paths):

        for i, file_path in enumerate(file_paths):
            print 'loading file ', i, 'out of ', len(file_paths)
            self._training_events += rp.open_event_file(file_path)
            self._training_labels += self.get_labels(file_path)

        self._training_features = self.get_features(self._training_events)



    def get_features(self, events):
        features = np.empty((len(events), self.num_features))



        for i, event in enumerate(events):
            data = (event._baseline[1] - event._data[:,1])/event._baseline[1]
            interp = scipy.interpolate.interp1d(range(event._data.shape[0]), data)
            for j in range(self.num_points):
                x = j*(1.*(data.shape[0]-1)/self.num_points)
                features[i,j] = interp(x)


        print 'feature matrix shape:', features.shape

        return features

    def get_labels(self, file_path):
        labels = []


        f = open(file_path, 'r')
        reader = csv.reader(f, delimiter = '\t')
        row = 5
        try:
            while row:
                row = reader.next()
                if row[0] == 'event#':
                    labels.append(row[-1])
        except:
            pass
        return labels


    def train_model(self, training_features, training_labels):
        self._model = LogisticRegression(fit_intercept = True)

        self._model.fit(training_features, training_labels)


        return

    def save_model(self, output_file_path):
        print 'output_file_path = ', output_file_path
        joblib.dump(self._model, output_file_path)
        return

    def make_predictions(self, features):
        predictions = self._model.predict(features)
        return predictions

    def make_predictions_proba(self, features):
        predictions = self._model.predict_proba(features)
        return predictions

    def load_model(self, model_file_path):
        self._model = joblib.load(model_file_path)
        return
