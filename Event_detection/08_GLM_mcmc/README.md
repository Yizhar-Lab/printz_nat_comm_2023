## Transformation and GLMs

The code in this folder does two things:
- Transforms the predictors in pretorTable.txt using a robust ZCA, to whiten it.
- Run a Horseshoe prior GLM to find the most relevant features with high predictive power.

---

Python environment installed with conda.

Use environment.yml to create the python enviroment in miniconda3. 

```
conda env create -f environment.yml
```

If running on Windows or Mac the portable environment might work better.

```
conda env create -f environment_portable.yml
```



## Running this code
To make the transformed versions of the data run :

```
python make_transformations.py
```
This will create the transformed versions of the predictor variables in the transformed_dfs folder.


---
To run the GLM run 
```
python run_on_transformed.py
```
This should create the required plots in the folder plots2

