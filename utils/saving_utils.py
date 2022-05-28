import os
import pickle
from typing import Dict
import pandas as pd
import yaml


def save_experiment_outputs(outputs: Dict, save_path: str):
    """

    Args:
        outputs: dict of experiment outputs.
        save_path: path to experiment dir.


    """
    if "dfs" in outputs.keys():
        save_dataframes(outputs["dfs"], save_path)
    if "pkl" in outputs.keys():
        save_pkls(outputs["pkl"], save_path)


def save_pkls(d: Dict, save_path):
    """
    gets a dict of object to save and saves each one of them as a pickle.

    Args:
        save_path:  where to save the pkls.
        d:  dict of object to save.

    """

    os.makedirs(os.path.join(save_path, "results", "pkl"), exist_ok=True)
    path = os.path.join(save_path, "results", "pkl")

    for obj_name, obj in d.items():
        pkl_path = os.path.join(path, obj_name + ".pickle")
        save_pkl(obj, pkl_path)


def save_pkl(obj, save_path: str):
    """

    Args:
        obj: any python object.
        save_path: path where obj will be save at.

    """

    with open(save_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_dataframes(d: Dict[str, pd.DataFrame], save_path):
    """
    gets a dict of DataFrames to save and saves each one of them as a csv or parquet (depends on its size).

    Args:
        save_path: where to save the dataframes.
        d: dict of DataFrames to save.

    """
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)
    path = os.path.join(save_path, "results")

    for k, df in d.items():
        save_df(df, os.path.join(path, k))


def save_df(df: pd.DataFrame, save_path: str):
    """

    Args:
        df: a DataFrame.
        save_path: path where df will be save at.

    """
    # if df.memory_usage(index=True).sum() / 1e+9 > 0.5:
    #     df.to_parquet(save_path + '.parquet')
    # else:
    df.to_csv(save_path + '.csv', index=False)


def save_args_as_config(args, save_path):
    """
    saves experiment args as a config yaml file.
    Args:
        args: experiment args.

    """
    os.makedirs(os.path.join(save_path, "config"), exist_ok=True)
    save_path = os.path.join(save_path, "config", "config.yaml")
    config = vars(args)
    save_yaml(config, save_path)


def save_yaml(config: Dict, save_path: str):
    """

    Args:
        config: a dictionary of configurations.
        save_path: defines where to save the config.

    """
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=True)


def create_dirs(args):
    """
    creates experiment directories if they don't exist.
    Args:
        args:  experiment args.

    """
    base_path = os.path.join(args.experiment_path, args.experiment_name)

    os.makedirs(os.path.join(base_path, args.train_dir), exist_ok=True)
    os.makedirs(os.path.join(base_path, args.validation_dir), exist_ok=True)
    os.makedirs(os.path.join(base_path, "logs"), exist_ok=True)
