from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, scale
from scipy.sparse import issparse

from dataset.gen_dataset import DATASET_FUNC, SynthConfig, ReadDataConfig, LibsvmConfig


class CustomDataset:
    """
    An abstract class that represents a Dataset
    """
    def __init__(self, dataset_name: str, sampling_seed: int, dataset_config: dataclass):
        self.features, self.labels = DATASET_FUNC[dataset_name](dataset_config)
        self.n = self.__len__()
        self.rng = np.random.default_rng(sampling_seed)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.labels)

    def get_dim(self):
        """

        :return: The number of features.
        """
        return self.features.shape[1]

    def normalize_features(self):
        """
        normalize the features such that the L2 norm of the features is 1.
        :return:
        """
        self.features = normalize(self.features, axis=1)

    def sample_data(self,
                    probabilities: np.array,
                    n_samples: int
                    ) -> Tuple:
        """
        samples the dataset by a probability vector.
        Args:
            probabilities:  a vector with a sampling probability for every index in the dataset.
            n_samples:  how many samples to return.

        Returns:  subset of the dataset that was sampled by the probabilities vector.

        """

        assert len(probabilities) == self.n, f"probabilities len: {len(probabilities)} !=  dataset len: {self.n}"

        indices = self.rng.choice(self.n, n_samples, p=probabilities)
        return self.features[indices], self.labels[indices], indices

    def get_data(self) -> Tuple:
        """

        Returns:  the full dataset.

        """
        return self.features, self.labels

    def dataset_to_df(self) -> pd.DataFrame:
        """

        Returns: a pd.DataFrame object of the features and the labels.
        example :
                feature_0	feature_1	feature_2	label
                0.13	    -0.05	    0.017       1
                -0.077	    -0.036	    0.10       -1

        """
        features, labels = self.get_data()

        return pd.concat(
            [
                pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])]),
                pd.DataFrame(labels, columns=["label"]),
            ],
            axis=1
        )


def maybe_to_arry(m):
    if not isinstance(m, np.ndarray):
        return m.toarray()
    else:
        return m

DATASET_CONFIGS = dict(
    Synth=SynthConfig,
    read_from_file=ReadDataConfig,
    libsvm=LibsvmConfig,
)