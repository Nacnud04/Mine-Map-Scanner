# Mine-Map-Scanner
Parses test from old mining claim maps (~1850-1980) and exports as a csv.

# Contents
* Checking python version
* Required libraries
* Running the program
* Settings and their functions
  * -i --image
  * -d --display
  * -dg --divide-grid
  * -de --divide-exact
  * -sa --start-angle
  * -fa --final-angle
  * -af --angle-fidelity
  * -lr --line-removal
  * -c --min-conf
  * -r --min-ratio
  * -mw --min-width
  * -mh --min-height
  * -v --view-type
  * -oh --output-height
  * -f --apply-filters
  * -oi --override-invert
  * -ir --intersection-requirement
  * -rc --result-consolidation
---
# Setting up environment
## Checking python version
This program requires 'python3'. Running this in 'python2' will not work. Python '3.10' is ideal.
You can check your python version by opening terminal typing the following:
````
python -V
````
    python3 -V

Which ever returns the proper version should be the command which the program is run from.

---
## Required libraries
Libraries are programs created by other people which other programs use to run higher level operations. This program requires quite a few of them.
### Preprocessing, Postprocessing, OCR
These libraries are in this github repository, and they were written along with the program. To include these the files 'PreProcessing.py', 'PostProcessing.py', and 'OCR.py' all need to be included in the same directory as 'main.py'
### The rest
```
The rest of the libraries include argparse, cv2, numpy, datetime, time, scipy, matplotlib
These can be installed with:
```
````
pip install argparse opencv-python
