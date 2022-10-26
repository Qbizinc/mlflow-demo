# Environment
--------

The repository has an `environmet.yml` file to install all the dependencies using conda:

```
conda env create -f environment.yml
```

the conda environment is called `mlflow_demo`.

There is also the file `requirements.txt` to install the libraries with another environment module.

# How to use

The notebook **mlflow_demo** has the code to run some examples to generate the data, train the model and log it. It is necessary to run>

1. Run Jupyter lab to acces to the notebook. The default listening at http://127.0.0.1:8888
```
> jupyter lab
```
2. Run the mlflow ui to visualize the experiments. The default listening at http://127.0.0.1:5000
```
> mlflow ui
```
