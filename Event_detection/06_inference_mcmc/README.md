## Running inference 

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

---


## Running this code
To run the inference, save the exports from the previous step in ```new_exports``` folder. The pairs to infer file contains all the cells for which the connectivity will be infered.

To run on a PC
```
python job_notebook_new_3_cells.py 0
```

To run on a high performance cluster, use the 
```
array_job_class.job
```
file as the job file on a LSF cluster.

## Concatenating the cluster outup

To combine the independent runs of the cluster into a single file use 

```
python concat_dfs.py
```
after setting the csvs varible to the path to the outpud csvs from the job.