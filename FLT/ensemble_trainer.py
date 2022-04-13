import sklearn
from sklearn.base import BaseEstimator
from local_expert import LocalExpert


__author__ = 'Emanuele'


class EnsembleTrainer(BaseEstimator):

    def __init__(self, base_estimator, centroid_picker, weigher_sampler):
        self.base_estimator = base_estimator
        self.centroid_picker = centroid_picker
        self.weigher_sampler = weigher_sampler

    def fit(self, n_experts, instances, labels):
        experts = []
        self.weigher_sampler.train(instances)
        for centroid in self.centroid_picker.pick(instances, labels, n_experts):
            expert = LocalExpert(sklearn.clone(self.base_estimator), self.weigher_sampler)
            expert.fit(instances, labels, centroid)
            experts.append(expert)


        # return expert
        return experts

    def predict(self):
        pass

    def classes_(self):
        pass

    def set_params(self, **params):
        pass