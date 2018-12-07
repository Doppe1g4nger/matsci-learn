import os

import matplotlib.pyplot as plt
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from matminer.datasets import load_dataset

from mslearn.pipeline import MatPipe


class Analytics:
    def __init__(self, predictive_model, feature_labels=None, dataset=None):
        if isinstance(predictive_model, MatPipe):
            self.predictive_model = predictive_model
            self.feature_labels = predictive_model.learner.features
            self.target = predictive_model.learner.fitted_target
            self.dataset = predictive_model.post_fit_df.drop([self.target],
                                                             axis=1)
        if feature_labels is not None:
            self.feature_labels = feature_labels
        if dataset is not None:
            self.dataset = dataset

        self.interpreter = Interpretation(self.dataset,
                                          feature_names=self.feature_labels)

        def predict_func(x):
            prediction = self.predictive_model.learner.predict(x, self.target)
            return prediction[self.target + " predicted"].values

        self.model = InMemoryModel(
            predict_func, examples=self.dataset
        )

    def get_feature_importance(self):
        return self.interpreter.feature_importance.feature_importance(
            self.model, progressbar=False
        )

    def plot_partial_dependence(self, feature_ids, save_plot=False,
                                show_plot=True):
        if not isinstance(feature_ids, list):
            feature_ids = [feature_ids]

        pretty_feature = " ".join(feature_ids[0].split("_"))
        pretty_target = " ".join(map(str.capitalize, self.target.split("_")))

        plt.rcParams.update({'font.size': 24})

        fig, ax = self.get_partial_dependence(feature_ids)
        ax.set_frame_on(False)

        ax.set_title("Model dependence on {}".format(pretty_feature))
        ax.set_ylabel("Average predicted {}".format(pretty_target))
        plt.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
        plt.legend().get_texts()[0].set_text(pretty_target)
        plt.tight_layout()

        if save_plot:
            plt.savefig("_".join([feature_ids[0], self.target, "pdp.png"]))

        if show_plot:
            plt.show()

    def get_partial_dependence(self, feature_ids):
        if not isinstance(feature_ids, list):
            feature_ids = [feature_ids]

        if len(feature_ids) != 1:
            raise ValueError("Error, this method does not yet support "
                             "computing plots for more than one feature at a "
                             "time")

        return self.interpreter.partial_dependence.partial_dependence(
            feature_ids, self.model, sample=False, progressbar=False
        )


if __name__ == '__main__':
    if not os.path.exists("tests/test_pipe.p"):
        df = load_dataset('elastic_tensor_2015')
        df = df[["formula", "K_VRH"]]
        df = df.rename({"formula": "composition"}, axis=1)

        fitted_pipeline = MatPipe().fit(df, "K_VRH")
        fitted_pipeline.save("tests/test_pipe.p")
    else:
        fitted_pipeline = MatPipe.load("tests/test_pipe.p")

    analyzer = Analytics(fitted_pipeline)

    x = analyzer.get_partial_dependence(analyzer.feature_labels[0])

    print(x)
