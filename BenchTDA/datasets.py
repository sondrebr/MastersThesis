"""Module for generating datasets for TDA benchmarking."""

import numpy as np
import os

from . import manifolds


""" Helper functions """


def _save(
    file: str,
    train_data: np.ndarray, train_labs: np.ndarray,
    val_data: np.ndarray, val_labs: np.ndarray,
    test_data: np.ndarray, test_labs: np.ndarray
) -> None:
    np.savez_compressed(
        file=file,
        train_data=train_data, train_labs=train_labs,
        val_data=val_data, val_labs=val_labs,
        test_data=test_data, test_labs=test_labs
    )


def _load(filepath: str) -> tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
    data_components = [
        "train_data", "train_labs",
        "val_data", "val_labs",
        "test_data", "test_labs"
    ]

    with np.load(filepath) as file:
        data_tup = tuple(file[comp] for comp in data_components)
    return data_tup


def _shuffle(
    x: np.ndarray,
    y: np.ndarray,
    gen: np.random.RandomState
):
    shuf_idx = gen.permutation(len(x))
    return x[shuf_idx], y[shuf_idx]


class _PointCloudGenerator:
    SAVE_DIR: str
    FILE_PATH: str
    short_name: str
    n_clouds: int
    n_points: int
    tv_noise: float
    test_noise: float
    split_ratios: tuple[float, float, float]
    normalized: bool
    seed: int
    cached: bool

    def __init__(
        self,
        short_name: str,
        n_clouds: int = 1000,
        n_points: int = 500,
        noise_scales: float | tuple[float, float] = 0.1,
        split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
        normalized: bool = True,
        seed: int = 42,
        cached=True,
        data_dir="BenchTDA_data/"
    ):
        self.n_clouds = n_clouds
        self.n_points = n_points
        self.split_ratios = split_ratios
        self.normalized = normalized
        self.seed = seed
        self.cached = cached

        if type(noise_scales) is float:
            self.tv_noise = noise_scales
            self.test_noise = noise_scales
        elif type(noise_scales) is tuple:
            if type(noise_scales[0]) is float and type(noise_scales[1]) is float:
                self.tv_noise, self.test_noise = noise_scales
        else:
            raise TypeError("noise_scales:", noise_scales, "Invalid type:", type(noise_scales))

        self.SAVE_DIR = (
            f'{data_dir}/{short_name}/'
            f'{n_clouds}-{n_points}-{str(noise_scales).replace(" ", "")}/'
            f'{seed}-{str(split_ratios).replace(" ", "")}-{"norm" if normalized else ""}/'
        )
        self.FILE_PATH = f'{self.SAVE_DIR}/data.npz'

    def _check_cache(self):
        # Load/gen.
        if (self.cached):
            os.makedirs(self.SAVE_DIR, exist_ok=True)

        # Go straight to data generation if caching is disabled
        data = None
        if self.cached:
            try:
                data = _load(self.FILE_PATH)
            except Exception:
                print("Unable to load data from cache. Generating...")
        return data

    def _standardize(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        test_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Translate and scale data to standard mean and variance."""
        mean, std = train_data.mean(axis=(0, 1)), train_data.std(axis=(0, 1))
        return (
            np.divide(np.subtract(train_data, mean), std),
            np.divide(np.subtract(val_data, mean), std),
            np.divide(np.subtract(test_data, mean), std)
        )

    def load(self) -> tuple[np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray]:
        """Load the dataset if self.cached is True and it is found in the cache. If not, generate the dataset.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training data, training labels, validation data, validation labels, test data, and test labels respectively.
        """
        if self.cached:
            data = self._check_cache()

        if data is None:
            gen = np.random.RandomState(seed=self.seed)
            data = self._generate(gen)
            if self.cached:
                _save(self.FILE_PATH, *data)

        return data

    def _generate(self, gen: np.random.RandomState):
        raise NotImplementedError()


""" Synthetic benchmarks """


class SphereTorusClassification(_PointCloudGenerator):
    """Generates a dataset of point clouds where one half of the point clouds are sampled from a 2-sphere, and the other half is sampled from a torus. The dataset is split into a training set, a validation set, and a test set."""

    def __init__(
        self,
        n_clouds: int = 1000,
        n_points: int = 100,
        noise_scales: float | tuple[float, float] = 0.1,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
        normalized: bool = True,
        seed: int = 42,
        cached=True,
        data_dir="BenchTDA_data/"
    ):
        """Initialize the class instance.

        Args:
            n_clouds (int, optional): The number of point clouds to generate. Defaults to 1000.
            n_points (int, optional): The number of points in each point cloud. Defaults to 100.
            noise_scales (float | tuple[float, float], optional): The noise scale(s) to use. If it is a tuple, the first number is used for the training and validation set, and the second number is used for the test set. Defaults to 0.1.
            split_ratios (tuple[float, float, float], optional): The split ratios to use for the training, validation, and test sets. Defaults to (0.6, 0.2, 0.2).
            normalized (bool, optional): Whether to normalize the datasets after generating. Defaults to True.
            seed (int, optional): The seed used for the random number generator. Defaults to 42.
            cached (bool, optional): Whether to cache the generated datasets. Defaults to True.
            data_dir (str, optional): The directory in which the datasets are cached. Only used if cached is True. Defaults to "BenchTDA_data/".
        """
        super().__init__(
            short_name="rn_st_c",
            n_clouds=n_clouds,
            n_points=n_points,
            noise_scales=noise_scales,
            split_ratios=split_ratios,
            normalized=normalized,
            seed=seed,
            cached=cached,
            data_dir=data_dir
        )

    def _generate(self, gen):
        data = []
        n_clouds_per_dataset: list[int] = []
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[0]))
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[1]))
        # Make sure sum(n_clouds_per_dataset) == self.n_clouds
        n_clouds_per_dataset.append(self.n_clouds - sum(n_clouds_per_dataset))

        for dataset_i, n_point_clouds in enumerate(n_clouds_per_dataset):
            noise_scale = self.test_noise if (dataset_i == 2) else self.tv_noise
            points = np.empty((n_point_clouds, self.n_points, 3))
            labs = np.empty((n_point_clouds, ), dtype=int)

            for pc_num in range(n_point_clouds):
                lab = pc_num % 2
                labs[pc_num] = lab

                if (lab == 0):
                    point_cloud = manifolds.nSphere(
                                    n=2,
                                    size=self.n_points,
                                    noise_scale=noise_scale*np.abs(gen.normal()),
                                    random_state=gen,
                    )
                else:
                    point_cloud = manifolds.uniform_torus(
                                    size=self.n_points,
                                    noise_scale=noise_scale*np.abs(gen.normal()),
                                    random_state=gen,
                    )

                point_cloud -= point_cloud.mean(axis=0)
                point_cloud /= point_cloud.std(axis=0)
                points[pc_num] = point_cloud

            data.extend(_shuffle(points, labs, gen))

        if self.normalized:
            data[0], data[2], data[4] = self._standardize(data[0], data[2], data[4])
        data = tuple(data)

        return data


class SphereGenusgTorusBinaryClassification(_PointCloudGenerator):
    """Generates a dataset of point clouds where one half of the point clouds are sampled from a 2-sphere, and the other half is sampled from a genus g torus of uniformly random genus 1 <= g <= 5. The dataset is split into a training set, a validation set, and a test set."""
    def __init__(
        self,
        n_clouds: int = 1000,
        n_points: int = 500,
        noise_scales: float | tuple[float, float] = 0.1,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
        normalized: bool = True,
        seed: int = 42,
        cached=True,
        data_dir="BenchTDA_data/"
    ):
        """Initialize the class instance.

        Args:
            n_clouds (int, optional): The number of point clouds to generate. Defaults to 1000.
            n_points (int, optional): The number of points in each point cloud. Defaults to 100.
            noise_scales (float | tuple[float, float], optional): The noise scale(s) to use. If it is a tuple, the first number is used for the training and validation set, and the second number is used for the test set. Defaults to 0.1.
            split_ratios (tuple[float, float, float], optional): The split ratios to use for the training, validation, and test sets. Defaults to (0.6, 0.2, 0.2).
            normalized (bool, optional): Whether to normalize the datasets after generating. Defaults to True.
            seed (int, optional): The seed used for the random number generator. Defaults to 42.
            cached (bool, optional): Whether to cache the generated datasets. Defaults to True.
            data_dir (str, optional): The directory in which the datasets are cached. Only used if cached is True. Defaults to "BenchTDA_data/".
        """
        super().__init__(
            short_name="rn_sggt_c",
            n_clouds=n_clouds,
            n_points=n_points,
            noise_scales=noise_scales,
            split_ratios=split_ratios,
            normalized=normalized,
            seed=seed,
            cached=cached,
            data_dir=data_dir
        )

    def _generate(self, gen):
        data = []
        n_clouds_per_dataset: list[int] = []
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[0]))
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[1]))
        # Make sure sum(n_clouds_per_dataset) == self.n_clouds
        n_clouds_per_dataset.append(self.n_clouds - sum(n_clouds_per_dataset))

        for i, n in enumerate(n_clouds_per_dataset):
            noise_scale = self.test_noise if (i == 2) else self.tv_noise
            points = np.empty((n, self.n_points, 3))
            labs = np.empty((n, ), dtype=int)

            for pc_num in range(n):
                lab = pc_num % 2
                labs[pc_num] = lab
                if (lab == 0):
                    point_cloud = manifolds.nSphere(
                        n=2,
                        size=self.n_points,
                        noise_scale=noise_scale*np.abs(gen.normal()),
                        random_state=gen
                    )
                else:
                    point_cloud = manifolds.genus_g_torus(
                        g=gen.randint(1, 6),
                        size=self.n_points,
                        cutoff_mod=np.tanh(gen.normal(0, 0.2)),
                        noise_scale=noise_scale*np.abs(gen.normal()),
                        random_state=gen
                    )

                point_cloud -= point_cloud.mean(axis=0)
                point_cloud /= point_cloud.std(axis=0)
                points[pc_num] = point_cloud

            data.extend(_shuffle(points, labs, gen))

        if self.normalized:
            data[0], data[2], data[4] = self._standardize(data[0], data[2], data[4])
        data = tuple(data)

        return data


class SphereGenusgTorusRegression(_PointCloudGenerator):
    """Generates a dataset of point clouds sampled from orientable manifolds of uniformly random genus g, 0 <= g <= 5. The dataset is split into a training set, a validation set, and a test set."""
    def __init__(
        self,
        n_clouds: int = 1000,
        n_points: int = 500,
        noise_scales: float | tuple[float, float] = 0.1,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
        normalized: bool = True,
        seed: int = 42,
        cached=True,
        data_dir="BenchTDA_data/"
    ):
        """Initialize the class instance.

        Args:
            n_clouds (int, optional): The number of point clouds to generate. Defaults to 1000.
            n_points (int, optional): The number of points in each point cloud. Defaults to 100.
            noise_scales (float | tuple[float, float], optional): The noise scale(s) to use. If it is a tuple, the first number is used for the training and validation set, and the second number is used for the test set. Defaults to 0.1.
            split_ratios (tuple[float, float, float], optional): The split ratios to use for the training, validation, and test sets. Defaults to (0.6, 0.2, 0.2).
            normalized (bool, optional): Whether to normalize the datasets after generating. Defaults to True.
            seed (int, optional): The seed used for the random number generator. Defaults to 42.
            cached (bool, optional): Whether to cache the generated datasets. Defaults to True.
            data_dir (str, optional): The directory in which the datasets are cached. Only used if cached is True. Defaults to "BenchTDA_data/".
        """
        super().__init__(
            short_name="rn_sggt_r",
            n_clouds=n_clouds,
            n_points=n_points,
            noise_scales=noise_scales,
            split_ratios=split_ratios,
            normalized=normalized,
            seed=seed,
            cached=cached,
            data_dir=data_dir
        )

    def _generate(self, gen):
        data = []
        n_clouds_per_dataset: list[int] = []
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[0]))
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[1]))
        # Make sure sum(n_clouds_per_dataset) == self.n_clouds
        n_clouds_per_dataset.append(self.n_clouds - sum(n_clouds_per_dataset))

        for i, n in enumerate(n_clouds_per_dataset):
            noise_scale = self.test_noise if (i == 2) else self.tv_noise
            points = np.empty((n, self.n_points, 3))
            labs = np.empty((n, ), dtype=int)

            for pc_num in range(n):
                lab = gen.randint(6)
                labs[pc_num] = lab
                if (lab == 0):
                    point_cloud = manifolds.nSphere(
                        n=2,
                        size=self.n_points,
                        noise_scale=noise_scale*np.abs(gen.normal()),
                        random_state=gen
                    )
                else:
                    point_cloud = manifolds.genus_g_torus(
                        g=lab,
                        size=self.n_points,
                        cutoff_mod=np.tanh(gen.normal(0, 0.2)),
                        noise_scale=noise_scale*np.abs(gen.normal()),
                        random_state=gen
                    )

                point_cloud -= point_cloud.mean(axis=0)
                point_cloud /= point_cloud.std(axis=0)
                points[pc_num] = point_cloud

            data.extend(_shuffle(points, labs, gen))

        if self.normalized:
            data[0], data[2], data[4] = self._standardize(data[0], data[2], data[4])
        data = tuple(data)

        return data


class PowerSphericalRegression(_PointCloudGenerator):
    """Generates a dataset of point clouds sampled using the Power Spherical distribution. The dataset is split into a training set, a validation set, and a test set."""
    def __init__(
        self,
        n_clouds: int = 1000,
        n_points: int = 100,
        noise_scales: float | tuple[float, float] = 0.1,
        split_ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
        normalized: bool = True,
        seed: int = 42,
        cached=True,
        data_dir="BenchTDA_data/"
    ):
        """Initialize the class instance.

        Args:
            n_clouds (int, optional): The number of point clouds to generate. Defaults to 1000.
            n_points (int, optional): The number of points in each point cloud. Defaults to 100.
            noise_scales (float | tuple[float, float], optional): The noise scale(s) to use. If it is a tuple, the first number is used for the training and validation set, and the second number is used for the test set. Defaults to 0.1.
            split_ratios (tuple[float, float, float], optional): The split ratios to use for the training, validation, and test sets. Defaults to (0.6, 0.2, 0.2).
            normalized (bool, optional): Whether to normalize the datasets after generating. Defaults to True.
            seed (int, optional): The seed used for the random number generator. Defaults to 42.
            cached (bool, optional): Whether to cache the generated datasets. Defaults to True.
            data_dir (str, optional): The directory in which the datasets are cached. Only used if cached is True. Defaults to "BenchTDA_data/".
        """
        super().__init__(
            short_name="rn_psc_r",
            n_clouds=n_clouds,
            n_points=n_points,
            noise_scales=noise_scales,
            split_ratios=split_ratios,
            normalized=normalized,
            seed=seed,
            cached=cached,
            data_dir=data_dir
        )

    def _generate(self, gen: np.random.RandomState):
        data = []
        n_clouds_per_dataset: list[int] = []
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[0]))
        n_clouds_per_dataset.append(int(self.n_clouds * self.split_ratios[1]))
        # Make sure sum(n_clouds_per_dataset) == self.n_clouds
        n_clouds_per_dataset.append(self.n_clouds - sum(n_clouds_per_dataset))

        for i, no_pcs in enumerate(n_clouds_per_dataset):
            noise_scale = self.test_noise if (i == 2) else self.tv_noise
            points = np.empty((no_pcs, self.n_points, 3))
            labs = abs(gen.normal(scale=2, size=no_pcs))
            directions = manifolds.nSphere(n=2, size=no_pcs, random_state=gen)

            for pc_num, (dir, conc) in enumerate(zip(directions, labs)):
                point_cloud = manifolds.power_spherical(
                    dim=3,
                    size=self.n_points,
                    direction=dir,
                    concentration=conc,
                    noise_scale=noise_scale*np.abs(gen.normal()),
                    random_state=gen
                )

                point_cloud -= point_cloud.mean(axis=0)
                point_cloud /= point_cloud.std(axis=0)
                points[pc_num] = point_cloud

            # Random labels - no need to shuffle
            data.append(points)
            data.append(labs)

        if self.normalized:
            data[0], data[2], data[4] = self._standardize(data[0], data[2], data[4])
        data = tuple(data)

        return data
