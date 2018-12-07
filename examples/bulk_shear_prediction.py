import os
from math import sqrt
from time import sleep

from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from matminer.utils.io import load_dataframe_from_json
# from matminer.datasets.convenience_loaders import load_tehrani_superhard_mat

from automatminer.pipeline import MatPipe
from automatminer.analytics import Analytics

"""
This script reproduces the results found by Tehrani et al in the following:
pubs.acs.org/doi/suppl/10.1021/jacs.8b02717/suppl_file/ja8b02717_si_001.pdf

We first train an SVM model as described in the above paper using identical
training data and model parameters. We then compare their results to the 
performance of our own model auto generated using the matsci-learn pipeline 
with features generated using the matminer software package.
"""


def load_tehrani_superhard_mat(data="all"):
    if data not in {"all", "engineered_features", "basic_descriptors"}:
        raise ValueError(f"Data description type {data} not recognized")

    df = load_dataframe_from_json("tehrani_superhard_mat.json.gz")

    if data == "engineered_features":
        df = df.drop(["formula", "composition",
                      "structure", "initial_structure",
                      "material_id"], axis=1)
    elif data == "basic_descriptors":
        df = df.drop([thing for thing in df.columns if thing not in {
            "composition", "structure", "bulk_modulus", "shear_modulus",
        }], axis=1)

    return df


if __name__ == '__main__':
    # TEHRANI REPRODUCTION
    # Load in training data and separate out features from targets
    df = load_tehrani_superhard_mat(data="engineered_features")

    target_labels = ["bulk_modulus", "shear_modulus"]
    # Convert to numpy array to work easier with sklearn
    features = df.drop(target_labels, axis=1).values

    # Define cross validation scheme, pre-processing pipeline and model to test
    k_folds = KFold(n_splits=10, shuffle=True, random_state=42)
    scaler = preprocessing.StandardScaler()
    normalizer = preprocessing.Normalizer()
    model = SVR(gamma=.01, C=10)

    # For each target train a model and evaluate with cross validation
    for target_label in target_labels:
        # Convert to numpy array to work easier with sklearn
        target = df[target_label].values

        # Hold onto these to calculate average performance
        r2_scores = []
        rmse_scores = []
        # Hold onto these to graph predicted vs. actual
        prediction_list = []
        target_list = []

        for train_index, test_index in tqdm(
                k_folds.split(features, target),
                desc="Cross validation progress for {}".format(target_label),
                total=10
        ):
            train_features = features[train_index]
            train_target = target[train_index]

            test_features = features[test_index]
            test_target = target[test_index]

            # Normalization is unique to each sample, doesn't need fit
            train_features = normalizer.transform(train_features)
            test_features = normalizer.transform(test_features)

            # Fit scaling on training data, don't see training data
            scaler.fit(train_features)
            train_features = scaler.transform(train_features)
            test_features = scaler.transform(test_features)

            model.fit(train_features, train_target)

            prediction = model.predict(test_features)
            r2_scores.append(r2_score(test_target, prediction))
            rmse_scores.append(sqrt(mean_squared_error(test_target,
                                                       prediction)))

            prediction_list += [item for item in prediction]
            target_list += [item for item in test_target]

        # Set up bounds for a pretty plot of prediction vs. ground truth
        target_and_prediction = np.concatenate((prediction_list, target_list))
        bounds = [min(target_and_prediction) - 20,
                  max(target_and_prediction) + 20]
        plt.xlim(bounds)
        plt.ylim(bounds)

        # Target on x and prediction on y means values lower than y=x
        # are under-prediction and above are over-prediction
        plt.scatter(target_list, prediction_list)

        # Plot line representing perfect predictor
        x = np.linspace(bounds[0], bounds[1], 1000)
        plt.plot(x, x, color="black")

        # Plot labels
        plt.title("{} actual vs. predicted".format(target_label))
        plt.ylabel("Cross validated predictions for {}".format(target_label))
        plt.xlabel("Actual values for {}".format(target_label))

        plt.show()

        # Output average accuracies across CV runs
        tqdm.write("RMSE for {}: {}".format(
            target_label, sum(rmse_scores) / len(rmse_scores)
        ))
        tqdm.write("R^2 for {}: {}".format(
            target_label, sum(r2_scores) / len(r2_scores)
        ))

        sleep(1)

    # COMPARE TO MATBENCH
    df = load_tehrani_superhard_mat(data="basic_descriptors")

    bulk_train = df.drop(["shear_modulus"], axis=1)

    if not os.path.exists("bulk_modulus_test_pipe.p"):
        fitted_pipeline = MatPipe().fit(bulk_train, "bulk_modulus")
        fitted_pipeline.save("bulk_modulus_test_pipe.p")
    else:
        fitted_pipeline = MatPipe.load("bulk_modulus_test_pipe.p")

    analyzer = Analytics(fitted_pipeline)
    print(analyzer.get_feature_importance())
    feats = list(reversed(analyzer.get_feature_importance().index))

    for feat in feats:
        analyzer.plot_partial_dependence(feat, save_plot=True, show_plot=False)

    shear_train = df.drop(["bulk_modulus"], axis=1)

    if not os.path.exists("shear_modulus_test_pipe.p"):
        fitted_pipeline = MatPipe().fit(shear_train, "shear_modulus")
        fitted_pipeline.save("shear_modulus_test_pipe.p")
    else:
        fitted_pipeline = MatPipe.load("shear_modulus_test_pipe.p")

    analyzer = Analytics(fitted_pipeline)
    feats = list(reversed(analyzer.get_feature_importance().index))

    for feat in feats:
        analyzer.plot_partial_dependence(feat, save_plot=True, show_plot=False)
