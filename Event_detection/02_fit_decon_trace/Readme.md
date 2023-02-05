# Code to run automated fitting of events using the output of deconvolution.



## Python environment
This step can be run in the same python environment as 01_decon_oasis


## Set up

Make the names files from the directory containing all the segment  .npz files using the commmand as follows:
```
ls -d -1 segments_File_2018_11_01_0025_Cell3/* > names_for_fit.txt
```

Here for demonstration 30 files from the previous step have been included. 

## running

### PC
To run all the files on a PC set the `step` variable to 1 
in line 37 in `array_fit.py`

Then run 
```
python array_fit.py 0
```

### High performance cluster
To run the parallel fit on a high performance cluster set the `step` variable to 1000
in line 37 in `array_fit.py`

Then each job will take the job number, from 1-1000 as the argument. For example the 200th job

```
python array_fit.py 200
```

## Output
The output is in the Results folder. Each job has a folder with `fit.csv`, and any skipped files (if at all) in the `skipped` folder.