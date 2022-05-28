import os
import pickle
import pandas as pd
from typing import Tuple
import numpy as np
import libsvmdata
from dataclasses import dataclass
from sklearn.svm import LinearSVC


@dataclass
class SynthConfig:
    dataset_name: str = "Synth"
    dataset_size: int = 500
    dim: int = 200
    dataset_seed: int = 3
    seperable: bool = False
    mean_constraint: Tuple = (0.5, 0.5)


def gen_binary_synthetic_dataset(config: SynthConfig) -> Tuple:
    """
    generates synthetic dataset by sampling from N(\mu_1, \sigma^2_1) and  N(\mu_2, \sigma^2_2).

    """

    rng = np.random.default_rng(config.dataset_seed)

    counts = np.round(np.array([0.5, 0.5]) * config.dataset_size).astype("int")
    cov = np.identity(config.dim)

    if config.seperable:
        mean = rng.normal(size=config.dim)

        pos_gaus = rng.multivariate_normal(mean, cov, counts[0])
        neg_gaus = rng.multivariate_normal(-mean, cov, counts[0])

    else:
        assert config.dim < config.dataset_size, "wrong config! N < d"
        score = 1
        while score ==1:
            pos_mean = gen_const_mean_vector(config.mean_constraint[0], config.dim, rng)
            pos_gaus = rng.multivariate_normal(pos_mean, cov, counts[0])

            neg_mean = gen_const_mean_vector(config.mean_constraint[1], config.dim, rng)
            neg_gaus = rng.multivariate_normal(neg_mean, cov, counts[0])

            features = np.concatenate((pos_gaus, neg_gaus))
            labels = np.repeat(np.array([1, -1]), counts)

            clf = LinearSVC(penalty='l2', loss='hinge', fit_intercept=False).fit(features, labels)
            score = clf.score(features, labels)

            config.mean_constraint = (config.mean_constraint[0] / 2, config.mean_constraint[1] / 2)
            cov *= 2

    return (
        np.concatenate((pos_gaus, neg_gaus)),
        np.repeat(np.array([1, -1]), counts)
    )


def gen_const_mean_vector(mean_const, dim, rng):
    vec = rng.normal(size=dim)
    return vec * mean_const / np.linalg.norm(vec)


@dataclass
class ReadDataConfig:
    dataset_root: str = "/data"
    dataset_name: str = "cifar10_vitb32_clip_features"
    dataset_size: int = None
    is_pickle: bool = True
    dataset_seed: int = 3
    train: bool = True


def read_from_file(config: ReadDataConfig) -> Tuple:
    rng = np.random.default_rng(config.dataset_seed)

    if config.train:
        folder_type = "train"
    else:
        folder_type = "test"

    if config.is_pickle:
        dataset_path = os.path.expanduser('~') + os.path.join(config.dataset_root, config.dataset_name, f'{folder_type}.pickle')
        features, labels = from_pickle(dataset_path)

    else:
        dataset_path = os.path.expanduser('~') + os.path.join(config.dataset_root, config.dataset_name + '.csv')
        dataset = read_pandas(dataset_path)
        features = dataset[[col for col in dataset.columns if "feature" in col]].values
        labels = dataset.label.values

    if config.dataset_size and config.dataset_size <= len(features):
        ind = rng.integers(0, len(features) - 1, config.dataset_size)
        features, labels = features[ind], labels[ind]

    return features, labels


def read_pandas(path) -> pd.DataFrame:
    if path.endswith(".csv"):
        df = pd.read_csv(path)

    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)

    else:
        raise ValueError("unsupported dataset file type")
    return df


def from_pickle(dataset_path):
    file_path = os.path.join(dataset_path)
    with open(file_path, 'rb') as f:
        ldd = pickle.load(f)
    features = ldd['features']
    labels = ldd['targets']
    return features, labels


@dataclass
class LibsvmConfig:
    dataset_name: str = "a9a"
    dataset_size: int = None
    dataset_root: str = None
    dataset_seed: int = 3
    train: bool = True


def libsvm_dataset(config: LibsvmConfig) -> Tuple:
    rng = np.random.default_rng(config.dataset_seed)

    if not config.train:
        config.dataset_name += "_test"

    if config.dataset_root is not None:
        libsvmdata.datasets.DATA_HOME = os.path.join(config.dataset_root, 'libsvm')

    features, labels = libsvmdata.fetch_libsvm(config.dataset_name)
    labels = labels.astype(int)

    if config.dataset_name == "covtype.binary":
        labels = convert_to_binary(labels)

    if config.dataset_size and config.dataset_size <= len(features):
        ind = rng.integers(0, len(features) - 1, config.dataset_size)
        features, labels = features[ind], labels[ind]
    return features, labels


def convert_to_binary(labels):
    if pd.Series(labels).nunique() > 2:
        labels = (labels > (labels.max() - 1) / 2).astype(int)
    else:
        labels = (labels >= labels.max()).astype(int)
    return np.where(labels == 0, -1, labels)


DATASET_FUNC = dict(
    Synth=gen_binary_synthetic_dataset,
    read_from_file=read_from_file,
    libsvm=libsvm_dataset,
)