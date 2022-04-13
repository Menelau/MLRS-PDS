import pickle
import numpy
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV

from core import matrix_utils





class EOLE(object):

    def __init__(self, n_experts, ensemble_trainer, preprocessor, use_probs, use_competences,data_name):
        self.n_experts = n_experts
        self.ensemble_trainer = ensemble_trainer
        self.preprocessor = preprocessor
        self.use_probs = use_probs
        self.use_competences = use_competences
        self.labels = None
        self.experts = None
        self.centroids = None
        self.data_name=data_name


    def fit(self, instances, labels):
        self.labels, labels = numpy.unique(labels, return_inverse=True)
        if self.preprocessor is not None:
            c = self.preprocessor.fit_transform(instances)

        self.experts = self.ensemble_trainer.fit(self.n_experts, instances, labels)

        self.centroids = numpy.asarray([expert.centroid for expert in self.experts])

        pkl_filename = self.data_name + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(self.experts, file)

    def predict(self, instances):
        return matrix_utils.prediction_matrix(self.predict_probs(instances), self.labels)

    def predict_probs(self, instances):
        if self.preprocessor is not None:
            instances = self.preprocessor.transform(instances)

        competence_matrix = matrix_utils.competence_matrix(instances, self.experts)
        probability_matrix = matrix_utils.probability_matrix(instances, self.experts, len(self.labels))

        # Sort the probability matrix in decreasing competence order
        indices = matrix_utils.order(competence_matrix)
        competence_matrix = matrix_utils.sort_matrix(competence_matrix, indices)
        probability_matrix = matrix_utils.sort_matrix(probability_matrix, indices)

        if not self.use_probs:
            probability_matrix = matrix_utils.sharpen_probability_matrix(probability_matrix)
        if self.use_competences:
            ensemble_probs = matrix_utils.partial_row_average_matrix(probability_matrix, competence_matrix)
        else:
            ensemble_probs = matrix_utils.partial_row_average_matrix(probability_matrix)
        return ensemble_probs

