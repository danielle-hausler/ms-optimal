import os
import sys
from dataclasses import dataclass, asdict
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from algorithms.AGD import AGD, AGDConfig
from algorithms.gradient_descent import gradient_descent, GDConfig
from algorithms.ms_algorithm import opt_ms_algorithm, ms_algorithm, MSBisectionConfig, MSNoBisectionConfig
from algorithms.acr import ACRConfig, acr_algorithm
from algorithms.solvers import minimize_obj
from algorithms.adaptive_acr import AdaptiveACRConfig, adaptive_acr_algorithm
from algorithms.scipy_optimizer import scipy_optimizer, ScipyOptimizerConfig
from algorithms.newton import newton_method, NewtonMethodConfig
from dataset.custom_dataset import CustomDataset, DATASET_CONFIGS
from dataset.gen_dataset import LibsvmConfig
from objective.chain_function import GenChain
from objective.logistic_regression import LogisticRegression
from utils.saving_utils import save_df


@dataclass
class Config:
    # experiment details
    description: str = ""

    # saving args
    output_dir: str = "../experiments_results"
    save_results: bool = True

    # dataset args
    dataset: str = "libsvm"
    sampling_seed: int = 4
    dataset_config: dataclass = LibsvmConfig(dataset_name="a9a")
    normalize_features: bool = True

    # objective args
    objective_type: str = "logistic_regression"
    regularization: float = 0.0
    dim: int = 200 # only for gen_chain objective
    iteration_budget: int = 250

    # algorithm config
    algorithm: str = "opt_ms_algorithm"
    algorithm_config: dataclass = MSNoBisectionConfig()

    # logger
    logger_level: str = "INFO"

    # test
    test_size: float = 0


@logger.catch(onerror=lambda _: sys.exit(1))
def run_experiment(config: Config):
    logger.level(config.logger_level)
    logger.info(f"running experiment: {config}")

    if config.save_results:
        logs_dir = os.path.join(config.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        logger.add(os.path.join(logs_dir, "logs_file.log"))

    assert config.dataset in DATASET_CONFIGS.keys(), f"Unsupported dataset name:{config.dataset}"
    dataset = CustomDataset(config.dataset, config.sampling_seed, config.dataset_config)
    if config.normalize_features:
        dataset.normalize_features()
    features, labels = dataset.features, dataset.labels

    if config.test_size > 0:
        logger.info(f"test_size >0, trying to load test-set")
        try:
            if 'train' in asdict(config.dataset_config).keys():
                config.dataset_config.train = False
            test_dataset = CustomDataset(config.dataset, config.sampling_seed, config.dataset_config)
            test_features, test_labels = test_dataset.features, test_dataset.labels
            logger.info(f"loaded test-set")
        except:
            logger.info(f"test-set loading failed, splitting train set")
            features, test_features, labels, test_labels = train_test_split(
            dataset.features, dataset.labels, test_size=config.test_size, random_state=config.sampling_seed
            )

    else:
        logger.info(f"test-set size = 0")
        test_features, test_labels = np.empty(0), np.empty(0)

    if config.objective_type == "logistic_regression":
        assert len(np.unique(labels)) == 2

    assert config.objective_type in OBJECTIVE_CLS.keys(), f"Unsupported objective type: {config.objective_type}"
    if config.objective_type in ["gen_chain"]:
        obj = OBJECTIVE_CLS[config.objective_type](config.dim)
    else:
        obj = OBJECTIVE_CLS[config.objective_type](
        features, labels, test_features, test_labels, config.objective_type, regularization=config.regularization)

    if 'lambda_func' in asdict(config.algorithm_config).keys() and config.algorithm_config.lambda_func == "lower_A_bound":
        optimal_x = minimize_obj(obj, "sklearn", 300)
        setattr(config.algorithm_config, "norm_x_opt", int(np.linalg.norm(optimal_x)))

    x, outputs = run_algorithm(config.algorithm)(
        obj=obj,
        config=config.algorithm_config,
        iteration_budget=config.iteration_budget,
        save_results=config.save_results,
        checkpoints_path=config.output_dir
    )

    if config.save_results:
        save_df(outputs, config.output_dir)

    return x, outputs


def run_algorithm(algorithm):
    algorithms_func = dict(
        ms_algorithm=ms_algorithm,
        opt_ms_algorithm=opt_ms_algorithm,
        acr=acr_algorithm,
        adaptive_acr=adaptive_acr_algorithm,
        scipy_optimizer=scipy_optimizer,
        newton_method=newton_method,
        AGD=AGD,
        GD=gradient_descent,
    )

    assert algorithm in algorithms_func.keys(), f"Unsupported algorithm {algorithm}"

    return algorithms_func[algorithm]


ALG_CONFIG = dict(
    ms_algorithm=MSBisectionConfig,
    opt_ms_algorithm=MSNoBisectionConfig,
    acr=ACRConfig,
    adaptive_acr=AdaptiveACRConfig,
    scipy_optimizer=ScipyOptimizerConfig,
    newton_method=NewtonMethodConfig,
    AGD=AGDConfig,
    GD=GDConfig,
)


OBJECTIVE_CLS = dict(
    logistic_regression=LogisticRegression,
    gen_chain=GenChain
)


if __name__ == '__main__':
    run_experiment(Config())