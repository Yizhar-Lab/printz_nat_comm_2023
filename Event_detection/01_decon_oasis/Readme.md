##


## Python environment

Python environment installed with poetry : https://python-poetry.org/

inside this directory, run   
```
poetry install
```

More documentation can be found at https://python-poetry.org/docs/basic-usage/


The python environment can also be installed with the requirements.txt file. 

### installing oasis

In addition the `oasis` package needs to be installed from github, using

```
pip install git+https://github.com/j-friedrich/OASIS.git

```

---
## Running code

### names.txt
The `names.txt` file contains the names of all the files that need to be processes. 

### running
To run the detection on the a particular file in names.txt the index of the file needs to be passed to the script as follows; for the first file run :

```
python detect_with_oasis_and_save.py 0
```

For second file in `names.txt` :

```
python detect_with_oasis_and_save.py 1
```

and so on. 


Here only one file is included in the `DATA` folder as an example. 
