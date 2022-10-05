# Meta-learning Tree-structured Parzen Estimator

This package was used for the experiments of the paper `Multi-objective Tree-structured Parzen Estimator Meets Meta-learning`.

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
