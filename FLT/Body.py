import os
from scipy.spatial import distance
from sklearn import preprocessing, base
from eole_sklearn import EOLE
from sklearn.tree import DecisionTreeClassifier
from dataset_utils import ArffLoader
from experiment import Experiment
from centroid_picker import RandomCentroidPicker, AlmostRandomCentroidPicker
from ensemble_trainer import EnsembleTrainer
from eole import EOLE
from exponential_weigher import ExponentialWeigher
from generalized_bootstrap import GeneralizedBootstrap


def main():
    n_folds = 2   #5
    repetitions = 2    #20
    n_groups = 1
    group = 0
    ensembles = [
        ("random_forest", make_random_forest()),
        # ("bootstrap_eole_0100_01", make_eole(100, 1)),
        # ("small_eole_30_Nil", make_eole(30, 1)),
        # ("bootstrap_eole_1000_01", make_eole(1000, 1)),
        # ("bootstrap_eole_0100_05", make_eole(100, 5)),
        # ("bootstrap_eole_1000_05", make_eole(1000, 5)),
        # ("bootstrap_eole_0100_10", make_eole(100, 10)),
        # ("bootstrap_eole_1000_10", make_eole(1000, 10)),
        # ("bootstrap_eole_0100_20", make_eole(100, 20)),
        # ("bootstrap_eole_1000_20", make_eole(1000, 20))
    ]



    dataset_names = [
        "Adult_1",
        "Banana",
        # "Blood",
        # "Breast",
        # "CTG",
        # "Ecoli",
        # "Faults",
        # "German",
        # "GLASS",
        # "Haberman",
        # "Heart",
        # "ILPD",
        # "Ionosphere",
        # "Laryngeal1",
        # "Laryngeal3",
        # "Lithuanian",
        # "Liver",
        # "Magic",
        # "Mammographic",
        # "Monk"
        # "Phoneme",
        # "Pima",
        # "Segmentation",
        # "Sonar",
        # "Thyroid",
        # "Vehicle",
        # "splice",
        # "Vertebral",
        # "WDVG",
        # "Weaning"
        # "Wine"
    ]
    for dataset_name in dataset_names:
    #for dataset_name in evaluation.dataset_names(n_groups, group):
        print("Start experiments on: {}".format(dataset_name))
        for ens_name, ensemble in ensembles:
            exp_name = "{}_{}".format(dataset_name, ens_name)
            if os.path.isfile("reports/{}.rep".format(exp_name)):
                print("Experiment {} already done, going to next one.".format(exp_name))
                continue
            print("Start experiment: {}".format(exp_name))
            experiment = Experiment(
                name=exp_name,
                ensemble=ensemble,
                dataset_loader=ArffLoader("evaluation/datasets/{}.arff".format(dataset_name)),
                n_folds=n_folds,
                n_repetitions=repetitions
                )
            report = experiment.run()
            report.dump("report/")



def make_random_forest():
    return EOLE(
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=RandomCentroidPicker(),
            weigher_sampler=GeneralizedBootstrap(sample_percent=100, weigher=ExponentialWeigher(precision=0, power=1))
            ),
        preprocessor=None,
        use_probs=False,
        use_competences=False
    )


def make_eole(sample_percent, precision):
    ensemble=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=AlmostRandomCentroidPicker(dist_measure=distance.chebyshev),
            weigher_sampler=GeneralizedBootstrap(
                sample_percent=sample_percent,
                weigher=ExponentialWeigher(precision=precision, power=1, dist_measure=distance.chebyshev))
                            )
    print("Ensemble", type(ensemble))
    return EOLE(
        n_experts=100,
        ensemble_trainer=EnsembleTrainer(
            base_estimator=DecisionTreeClassifier(max_features="auto"),
            centroid_picker=AlmostRandomCentroidPicker(dist_measure=distance.chebyshev),
            weigher_sampler=GeneralizedBootstrap(
                sample_percent=sample_percent,
                weigher=ExponentialWeigher(precision=precision, power=1, dist_measure=distance.chebyshev)
            )
        ),
        preprocessor=preprocessing.MinMaxScaler(),
        use_probs=True,
        use_competences=False
    )




if __name__ == "__main__":
    main()
