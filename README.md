# Meta-learning Tree-structured Parzen Estimator

This package was used for the experiments of the paper `Speeding up Multi-objective Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-structured Parzen Estimator`.

## Setup

This package requires python 3.8 or later version.
You can install the dependencies by:

```shell
$ conda create -n meta-learn-tpe python==3.8
$ pip install -r requirements.txt

# Create a directory for tabular datasets
$ mkdir ~/tabular_benchmarks
$ cd ~/tabular_benchmarks

# The download of HPOLib
$ cd ~/tabular_benchmarks
$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
$ tar xf fcnet_tabular_benchmarks.tar.gz
$ mv fcnet_tabular_benchmarks hpolib
```

## Running example
The data obtained in the experiments are reproduced by the following command:

```shell
$ ./run_experiment.sh -s 0 -d 19
```

# Citations

For the citation, use the following format:
```
@article{watanabe2023ctpe,
  title={Speeding up Multi-objective Non-hierarchical Hyperparameter Optimization by Task Similarity-Based Meta-Learning for the Tree-structured {P}arzen Estimator},
  author={S. Watanabe and N. Awad and M. Onishi and F. Hutter},
  journal={International Joint Conference on Artificial Intelligence},
  year={2023}
}
```
