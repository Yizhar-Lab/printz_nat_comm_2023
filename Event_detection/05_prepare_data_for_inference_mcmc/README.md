## Code to prepare data for inference



Python environment installed with poetry : https://python-poetry.org/

in side this directory, run   
```
poetry install
```

More documentation can be found at https://python-poetry.org/docs/basic-usage/

---

To run the code : 
```
python split_into_pairs.py
```

This code saves all the cells that need to be infered (having atleast one event in the evoked period etc.) in the 
`new_exports/pairs_to_infer` file.
It also exports the relevant events and their metadata in the `new_exports/EVOKED_DF` and `new_exports/META_DF` files.

The entire folder `new_exports` is used in the next step. 


---
Running load_MADs and load_PC_cells requires having all relevant the matlab data files in the DATA folder. 
