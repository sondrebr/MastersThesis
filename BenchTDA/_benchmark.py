"""Benchmarking tool."""

from copy import deepcopy
from itertools import product
from typing import Callable
from time import time_ns
import json

import numpy as np

from sklearn.metrics import accuracy_score, mean_squared_error

from . import datasets


DEFAULT_SEED = 42
DATA_SAVE_DIR = "BenchTDA_data/"

# TODO: Implement custom max in benchmark
MAX_VEC_PARAM_COMBINATIONS = 20
MAX_HYPERPARAM_COMBINATIONS = 20


"""
    Task structure:
        (
            {
                "long_name":            str,
                "short_name":           str,
                "data_type":            str,
                "task_type":            Literal["classification", "regression"],
                "dataloader":           function,
                "constant_parameters":  {
                                            "...": ...,
                                            (...)
                                        }
                "variable_parameters":  {
                                            "...": [...],
                                            (...)
                                        }
            },
            ...
        )
"""

N_CLOUDS = 1000
SPLIT_RATIOS = (0.6, 0.2, 0.2)
NOISE_SCALES = [0.0, 0.1, (0.1, 0.2), 0.2, (0.2, 0.4)]

TASKS = [
    {
        "long_name":            "Sphere/torus classification",
        "short_name":           "rn_st_c",
        "data_type":            "pc",
        "task_type":            "classification",
        "dataloader":           datasets.SphereTorusClassification,
        "constant_parameters":  {
                                    "n_clouds":     N_CLOUDS,
                                    "split_ratios": SPLIT_RATIOS,
                                    "normalized":   True,
                                    "cached":       True,
                                },
        "variable_parameters":  {
                                    "n_points":     [50, 100, 250],
                                    "noise_scales": NOISE_SCALES,
                                }
    },

    {
        "long_name":            "Sphere/genus g torus binary classification",
        "short_name":           "rn_sggt_c",
        "data_type":            "pc",
        "task_type":            "classification",
        "dataloader":           datasets.SphereGenusgTorusBinaryClassification,
        "constant_parameters":  {
                                    "n_clouds":     N_CLOUDS,
                                    "split_ratios": SPLIT_RATIOS,
                                    "normalized":   True,
                                    "cached":       True,
                                },
        "variable_parameters":  {
                                    "n_points":     [250, 500, 1000],
                                    "noise_scales": NOISE_SCALES,
                                }
    },

    {
        "long_name":            "Sphere/genus g torus genus regression",
        "short_name":           "rn_sggt_r",
        "data_type":            "pc",
        "task_type":            "regression",
        "dataloader":           datasets.SphereGenusgTorusRegression,
        "constant_parameters":  {
                                    "n_clouds":     N_CLOUDS,
                                    "split_ratios": SPLIT_RATIOS,
                                    "normalized":   True,
                                    "cached":       True,
                                },
        "variable_parameters":  {
                                    "n_points":     [250, 500, 1000],
                                    "noise_scales": NOISE_SCALES,
                                }
    },

    {
        "long_name":            "Power Spherical concentration regression",
        "short_name":           "rn_psc_r",
        "data_type":            "pc",
        "task_type":            "regression",
        "dataloader":           datasets.PowerSphericalRegression,
        "constant_parameters":  {
                                    "n_clouds":     N_CLOUDS,
                                    "split_ratios": SPLIT_RATIOS,
                                    "normalized":   True,
                                    "cached":       True,
                                },
        "variable_parameters":  {
                                    "n_points":     [50, 100, 250],
                                    "noise_scales": NOISE_SCALES,
                                }
    },
]


""" Helper functions """


def _generate_task_combinations(tasks):
    final_tasks = []
    for task in tasks:
        new_task = deepcopy(task)
        new_task["parameters"] = new_task["constant_parameters"]
        variable_params = new_task.get("variable_parameters", {})

        if len(variable_params) == 0:
            final_tasks.append(new_task)
            continue

        task_param_combos = [
            dict(zip(variable_params.keys(), combo))
            for combo in product(*variable_params.values())
        ]

        for combo in task_param_combos:
            new_task_combo = deepcopy(new_task)
            new_task_combo["parameters"].update(combo)
            final_tasks.append(new_task_combo)
    return final_tasks


def _generate_pipeline_combinations(fscs, phs, vecs):
    combinations = {}

    for fsc_dict in fscs:
        fsc_tups = [
            (fsc_dict["in"], *fsc_tup)
            for fsc_tup in fsc_dict["fns"].items()
        ]
        compat_ph_dicts = [
            ph_dict for ph_dict in phs
            if ph_dict["in"] == fsc_dict["out"]
        ]

        for ph_dict in compat_ph_dicts:
            ph_tups = ph_dict["fns"].items()
            vec_tups = []
            for vec_dict in vecs:
                if vec_dict["in"] == ph_dict["out"]:
                    vec_tups.extend(vec_dict["classes"].items())

            # No vec_tups, no combinations - skip
            if len(vec_tups) == 0:
                continue

            for fsc in fsc_tups:
                fsc_combos_dict = combinations.get(fsc, {})

                for ph in ph_tups:
                    ph_combos_list = fsc_combos_dict.get(ph, [])
                    ph_combos_list.extend(vec_tups)
                    fsc_combos_dict[ph] = ph_combos_list

                combinations[fsc] = fsc_combos_dict

    return combinations


# TODO: Implement no_max, used only if all_discrete
def _generate_param_combinations(
        params_dict,
        max_combinations,
        gen: np.random.RandomState,
        no_max=False
):
    final_combinations = []

    all_discrete = not any([key[1] == "continuous" for key in params_dict.keys()])
    if all_discrete:
        all_combinations = list(product(*params_dict.values()))
        n_combinations = len(all_combinations)

        # Shuffle and get at most max_combinations from all_combinations
        # Ensures no duplicates
        final_n_combinations = min(n_combinations, max_combinations)
        indices = gen.choice(n_combinations, final_n_combinations, replace=False)
        param_combinations = [all_combinations[i] for i in indices]

        param_names = [key[0] for key in params_dict.keys()]
        final_combinations = [
            dict(zip(param_names, param_values))
            for param_values in param_combinations
        ]
    else:
        for _ in range(max_combinations):
            d = {}
            for (param_name, cont_type), param_range in params_dict.items():
                if cont_type == "continuous":
                    d[param_name] = gen.uniform(param_range[0], param_range[1])
                else:
                    d[param_name] = param_range[gen.choice(len(param_range))]
            final_combinations.append(d)

    return final_combinations


def _get_variable_params_values(task: dict):
    return [(k, task["parameters"][k]) for k in task["variable_parameters"].keys()]


def _get_score_keys(task: dict, combo_names: tuple):
    """Get the hierarchy of the scores dict.

    Args:
        task: The current task dict (with the "parameters" key)
        combo_names: Tuple containing the FSC, PH and vec. names

    Returns:
        list[tuple]: List of (_name, key) tuples
    """
    combo_key = f'{combo_names[0]}, {combo_names[1]}, {combo_names[2]}'

    keys: list[tuple] = [("Task name", task["long_name"])]

    for var_param in task["variable_parameters"].keys():
        keys.append((str(var_param), str(task["parameters"][str(var_param)])))

    keys.append(("FSC, PH, Vec.", combo_key))

    return keys


def _set_score(
        scores: dict, task: dict, combo_names: tuple, perf_metric: str,
        fsc_ph_time: float, vec_time: float,
        best_val_score: float, test_score: float):
    """Update scores dictionary with the times and scores of the current pipeline.

    Args:
        scores (dict): The current scores dictionary
        task (dict): The current task
        combo_names (tuple): String tuple with the names of the current FSC/PH/vec. combination
        fsc_ph_time (float): Time taken for FSC and PH
        vec_time (float): Time taken for rest of task (vectorization, model selection, testing)
        perf_metric (str): The performance metric used for the task
        best_val_score (float): Best validation score from the model selection
        test_score (float): The test score of the selected vectorization parameters and model

    Returns:
        dict: The updated scores dictionary
    """
    key_tuples = _get_score_keys(task, combo_names)
    current_level = scores

    for level_name, key in key_tuples:
        _ = current_level.setdefault("_name", level_name)
        _ = current_level.setdefault(key, {})
        if level_name == "FSC, PH, Vec.":
            current_level[key][f'Validation {perf_metric}'] = best_val_score
            current_level[key][f'Test {perf_metric}'] = test_score
            current_level[key]["FSC + PH time (s)"] = fsc_ph_time
            current_level[key]["Vec. time (s)"] = vec_time
        else:
            current_level = current_level[key]

    return scores


def _score_present(scores: dict, task: dict, combo_names: tuple):
    key_tuples = _get_score_keys(task, combo_names)
    current_level = scores

    for _, key in key_tuples:
        if current_level.get(key) is None:
            return False
        else:
            current_level = current_level[key]

    return True


""" Implementation """


def benchmark(
    fscs: list[dict[str, str | dict[str, Callable]]],
    phs: list[dict[str, str | dict[str, Callable]]],
    vecs: list[dict[str, str | dict[str, type]]],
    seed: int = DEFAULT_SEED,
    data_dir: str = DATA_SAVE_DIR,
    force=False,
    verbose=False,
):
    """Benchmark TDA pipelines with compatible FSC/PH/Vec. combos.

    Args:
        fscs (list[dict[str, str  |  dict[str, Callable]]]): A list of dictionaries containing the FSC functions. Each dictionary must be formatted as:
        {
            "in": string representing the input data type of the function,
            "out": "string representing the output data type of the function",
            "fns": {
                "the function name": the function itself,
                (...)
            }
        }

        phs (list[dict[str, str  |  dict[str, Callable]]]): A list of dictionaries containing the PH functions. Each dictionary must be formatted as:
        {
            "in": string representing the input data type of the function,
            "out": "string representing the output data type of the function",
            "fns": {
                "the function name": the function itself,
                (...)
            }
        }

        vecs (list[dict[str, str  |  dict[str, type]]]): A list of dictionaries containing the vectorization classes. Each dictionary must be formatted as:
        {
            "in": string representing the input data type of the function,
            "classes": {
                "the class name": the class itself (not an instance),
                (...)
            }
        }

        seed (int, optional): The seed used for the random number generators. Defaults to 42.
        data_dir (str, optional): The data in which the generated datasets are cached. Defaults to "BenchTDA_data/".
        force (bool, optional): Whether to force reruns of previously run combinations. Defaults to False.
        verbose (bool, optional): Gives more verbose output, printing the highest validation score of each pipeline along with the associated parameters. Defaults to False.
    """
    def save(scores: dict):
        with open("./scores.json", "w") as file:
            json.dump(scores, file, indent=4)

    try:
        with open("./scores.json", "r") as file:
            scores = json.load(file)
    except Exception:
        print("'scores' file not found. Initializing...")
        scores = {}
        save(scores)

    # Pre-generate tasks wrt. parameter combinations
    final_tasks = _generate_task_combinations(TASKS)

    # Pre-generate pipeline combinations
    combinations = _generate_pipeline_combinations(fscs, phs, vecs)
    # Combine fscs and phs to reduce indentation
    fsc_tup_ph_tup_combinations = [
        (fsc_tup, ph_tup, vec_list)
        for (fsc_tup, ph_dict) in combinations.items()
        for (ph_tup, vec_list) in ph_dict.items()
    ]

    for i, task in enumerate(final_tasks):
        print(f'Current dataset ({i+1}/{len(final_tasks)}): {task["long_name"]}', end="")
        if task.get("variable_parameters", {}) != {}:
            print(*[f', {k}: {v}' for (k, v) in _get_variable_params_values(task)], sep="")
        else:
            print("")

        if task["task_type"] == "classification":
            perf_metric = "accuracy"
        elif task["task_type"] == "regression":
            perf_metric = "MSE"

        task_params = deepcopy(task["parameters"])
        task_params["seed"] = seed
        task_params["data_dir"] = data_dir

        dataloader = task["dataloader"](**task_params)

        for (
            (fsc_in, fsc_name, fsc_fn),
            (ph_name, ph_fn), vec_tup_list
        ) in fsc_tup_ph_tup_combinations:
            if fsc_in != task["data_type"]:
                continue

            # Skip if not forced and all vecs. have been run
            if not force:
                already_run_vecs = []
                for (vec_name, VecClass) in vec_tup_list:
                    if _score_present(scores, task, (fsc_name, ph_name, vec_name)):
                        already_run_vecs.append((vec_name, VecClass))
                if already_run_vecs == vec_tup_list:
                    continue

            # Load (cached) data for each ph function
            (X_train, y_train,
             X_val,   y_val,
             X_test,  y_test) = dataloader.load()

            # Start timer after loading data
            fsc_ph_start_time = time_ns()

            # To ensure reproducible results, FSC
            # and PH calls are given the seed in
            # case the functions use random values

            # Make combined train_val data and labels
            ph_X_train_val = ph_fn(
                fsc_fn(deepcopy(np.vstack((X_train, X_val))), seed),
                seed
            )
            y_train_val = np.hstack((y_train, y_val))

            ph_X_train = ph_fn(fsc_fn(X_train, seed), seed)
            ph_X_val = ph_fn(fsc_fn(X_val, seed), seed)
            ph_X_test = ph_fn(fsc_fn(X_test, seed), seed)

            # Stop fsc+ph timer here, store as _fsc_ph_time in scores.json
            fsc_ph_stop_time = time_ns()
            fsc_ph_time = (fsc_ph_stop_time - fsc_ph_start_time) / 1e9

            for (vec_name, VecClass) in vec_tup_list:
                # If run is not forced, check if task/pipeline
                # combination has been run previously
                if not force:
                    has_been_run = _score_present(scores, task, (fsc_name, ph_name, vec_name))
                    if has_been_run:
                        continue

                print(f'Testing {fsc_name} -> {ph_name} -> {vec_name}:')
                print(f'FSC + PH time: {fsc_ph_time}s')
                vec_instance = VecClass(seed)

                vec_param_ranges = vec_instance.vec_parameter_ranges
                hyperparam_ranges = vec_instance.hyperparameter_ranges

                # Reset gen for each run to account for skips/different orders
                gen = np.random.RandomState(seed)

                vec_param_combinations = _generate_param_combinations(
                    vec_param_ranges,
                    MAX_VEC_PARAM_COMBINATIONS,
                    gen
                )

                hyperparam_combinations = _generate_param_combinations(
                    hyperparam_ranges,
                    MAX_HYPERPARAM_COMBINATIONS,
                    gen
                )

                # Actual vec. work begins here - start timer
                val_scores = {}
                vec_time_start = time_ns()

                for vec_params in vec_param_combinations:
                    vec_instance.fit(deepcopy(ph_X_train), vec_params)

                    vec_X_train = vec_instance.transform(deepcopy(ph_X_train))
                    vec_X_val = vec_instance.transform(deepcopy(ph_X_val))

                    # Get ModelClass here in case vectorization parameters
                    # are applied at model level, like with PersLay
                    if task["task_type"] == "classification":
                        ModelClass = vec_instance.classifier
                    elif task["task_type"] == "regression":
                        ModelClass = vec_instance.regressor

                    for hyperparams in hyperparam_combinations:
                        model = ModelClass(hyperparams, seed)

                        # Training
                        model.fit(deepcopy(vec_X_train), deepcopy(y_train))

                        pred_y_val = model.predict(deepcopy(vec_X_val))
                        if perf_metric == "accuracy":
                            val_score = accuracy_score(y_val, pred_y_val)
                        elif perf_metric == "MSE":
                            val_score = mean_squared_error(y_val, pred_y_val)

                        key = (tuple(vec_params.items()), tuple(hyperparams.items()))
                        val_scores[key] = val_score

                # Best parameters w/ score
                if perf_metric == "accuracy":
                    best_run_params = max(val_scores, key=lambda k: val_scores[k])
                elif perf_metric == "MSE":
                    best_run_params = min(val_scores, key=lambda k: val_scores[k])
                (best_vec_params_tup, best_hyperparams_tup) = best_run_params
                best_score = val_scores[best_run_params]

                if verbose:
                    print(
                        f'Best validation {perf_metric}'
                        f' was {best_score} with parameters:'
                    )
                    for k, v in (best_vec_params_tup + best_hyperparams_tup):
                        print(f'{k}: {v}')

                # Testing
                vec_instance.fit(deepcopy(ph_X_train_val), dict(best_vec_params_tup))
                vec_X_train_val = vec_instance.transform(deepcopy(ph_X_train_val))
                vec_X_test = vec_instance.transform(deepcopy(ph_X_test))

                test_model = ModelClass(dict(best_hyperparams_tup), seed)
                test_model.fit(vec_X_train_val, deepcopy(y_train_val))

                pred_y_test = test_model.predict(vec_X_test)
                if perf_metric == "accuracy":
                    test_score = accuracy_score(y_test, pred_y_test)
                elif perf_metric == "MSE":
                    test_score = mean_squared_error(y_test, pred_y_test)

                vec_time_stop = time_ns()
                vec_time_taken = (vec_time_stop - vec_time_start) / 1e9
                print(
                    f'Test {perf_metric} for'
                    f' {fsc_name} -> {ph_name} -> {vec_name}'
                    f' was {test_score}.'
                )
                print(f'Time taken: {vec_time_taken}s')
                print("")

                scores = _set_score(
                    scores=scores, task=task, combo_names=(fsc_name, ph_name, vec_name),
                    perf_metric=perf_metric,
                    fsc_ph_time=float(fsc_ph_time), vec_time=vec_time_taken,
                    best_val_score=float(best_score), test_score=float(test_score),
                )

                save(scores)
