# Mine-Map-Scanner
Parses test from old mining claim maps (~1850-1980) and exports as a csv.

# Contents
* Checking python version
* Required libraries
* Setting up the directory.
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
This program requires `python3`. Running this in `python2` will not work. Python `3.10` is ideal.
You can check your python version by opening terminal typing the following:
````
python -V
````
    python3 -V

Which ever returns the proper version should be the command which the program is run from.
If neither return an updated enough python version, or both commands return errors python needs to be updated.

---
## Required libraries
Libraries are programs created by other people which other programs use to run higher level operations. This program requires quite a few of them.
### Preprocessing, Postprocessing, OCR
These libraries are in this github repository, and they were written along with the program. To include these the files `PreProcessing.py`, `PostProcessing.py`, and `OCR.py` all need to be included in the same directory as `main.py`
### pytesseract
This library is a neural network designed for detecting words. This is the heart of the program.  
To install run the following through commandline:
```
pip install pytesseract
```
### pyLSD
This library is for straight line detection. The main fork was only developed for python2. There is a fork [here](https://github.com/AndranikSargsyan/pylsd-nova)  
To install run the following through commandline:
```
pip install pylsd-nova
```
### The rest
The rest of the libraries include argparse, cv2, numpy, datetime, scipy, matplotlib.  
These can be installed through commandline with:
```
pip install argparse opencv-python numpy datetime scipy matplotlib
```

---
## Setting up the directory
For ease of use the best folder directory contains the following (bold are folders):  
* PreProcessing.py
* PostProcessing.py
* OCR.py
* main.py
* **Map Name**
  * map.jpg
The folder named **Map Name** will contain all of the exported data, as well as the files which are created during program operation.

---
## Running the program
Now that the directory has been set up and all of the libraries have been installed we can run the program.  
To do this enter commandline and enter the directory which was set up in the previous step.
### Changing directories in command line
To move up a directory type the following:
```
cd "DirectoryName"
```
To move down a directory type the following:
```
cd ..
```
Using this syntax, navigate to the directory created earlier
### Running the program
To run the program use the following command:
```
python main.py -i "Map Name/map.jpg" -c 30 -dg 3,3 -sa -50 -fa 50 -af 5 -r 2
```
There are **a lot** of input parameters/settings here. The only one which is required is the `-i` parameter which here is set to `"Map Name/map.jpg"`. The next section details each of these settings.

---
# Input parameters / settings
## -i --image
This is the only **required** input parameter. This parameter requires the path to the map being scanned. An example path is `"Map Directory/map.jpg"`. Additionally running through multiple directories also works, for example: `"Maps/Map1Directory/myfirstscannedmap.jpg"`. The folder with the map file in it is the one where all of he output files will be placed, as well as the working files (files created and then destroyed by the program).  
The input map should ideally be at as high of a resolution and clarity as possible. Performance will be negatively affected with map quality.
## -d --display
When the program is finished running, it pops up the image with red boxes outlining where it thinks text was detected. This image by default is the image specified by `-i`, however `-d` is used it will display the results on the image at the path specified by `-d`. The path for the image is defined the exact same way as it was for -i. The display image used should be at the same resolution as the image defined by `-i`.  
The optimal use case of this setting would be if the user added filters to the input image by hand using a photo editor, but then wanted to display the results on the original image. In this case they would define the filtered image with `-i` and the original image at `-d`. 
