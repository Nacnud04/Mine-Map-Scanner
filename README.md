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
  * -st --segmentation-type
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
  * -oi --override-invert
  * -ir --intersection-requirement
  * -rc --result-consolidation
* During the program
* Output/Understanding the Results

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
For ease of use, the best folder directory contains the following (bold are folders):  
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
cd "Directory Name"
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
## -st --segmentation-type
The image can be segmented by two different methods, either `"grid"` or `"contour"`. Grid segmentation splits the image by the grid defined by `-dg`. This works well with medium to large words. Contour segmentation tries to identify where words are and then create segments around those locations. This works ok for medium words and well for small words, however this is slightly slower.
## -dg --divide-grid
When running using **grid segmentation** this input is **required**. When running using **contouring segmentation** this input should not be used.  
This is the function which defines the grid. This setting is inputed as `rr,cc` where `rr = #rows` and `cc = #columns` 
The more segments the image is divided into (the greater rr or cc) the more likely the program will be to correctly find smaller text and it will have a greater confidence with larger words. This also comes with the downside of increasing the amount of noise which might be detected as words (for instance small specks or dust in the scan).  
While this can be set at 1,1 if the user chooses not to divide the image it is recommended to at least split the image into 2x2 however 3x3 will perform better although likely take longer. 
Visually this looks like the following:
![griddiagram](https://github.com/Nacnud04/Mine-Map-Scanner/blob/main/images/GridDiagram.png?raw=true "grid diagram")
Overlapping segments need to exist as just the original divisions will cut words in half.
This means that the amount of images scanned is:
```
#image segments = (rr*2-1)*(cc*2-1)
```
## -de --divide-exact
This is the exact same setting as `--divide-grid` except it divides the image at pixel coordinates. This is unstable, has likely broken and should not be used.
## -sa --start-angle
When every image is scanned it is rotated through a set of angles so that the text in the image is detected no matter what angle it is at. The angle starts at the angle defined by this setting and increases by increments defined by `--angle-fidelity` until the value defined by `--final-angle` is reached. This value should be tuned depending on the angles which the text is at as it is much more efficent to not scan images with text at off angles if possible. The angles work on the image as seen here:
![angleaction](https://github.com/Nacnud04/Mine-Map-Scanner/blob/main/images/AngleAction.png?raw=true "angle action")
## -fa --final-angle
This is the final angle to rotate the image to. As in, when `start angle + iterations * angle fidelity = final angle` the program stops scanning.
## -af --angle-fidelity
This is the amount of degrees which the angle at which the image is incremented by.  
This means that the amount of image segments the program needs to scan becomes the following:
```
images to scan = floor(((final angle - start angle) / angle fidelity) + 1) * (rr*2-1)*(cc*2-1))
```

## -lr --line-removal
This enables/disables line removal and can either be set to `True` or `False`. Line removal generally improves accuracy as nearby distractions (line) are detected and then covered over with the median color of the image. If lines are smaller, fainter or dashed it is less likely to detect them. Additionally this will not remove lines which go through words or underline words, however the program is usually able to detect words even if they are underlined or have a line crossing through them, though it does greatly hurt accuracy.
Line removal works by detecting lines based off of their gradient using pyLSD (python line segment detection). These segments are usually very broken up and need to be fused together. If these lines are not fused together then they often create worse results than if the lines weren't removed at all. The lines are fused by a processses which weights their closeness in distance, as well as angle and then fuses their furthest points. Larger segments have power in the weighting algorithm than smaller lines.
## -c --min-conf
This crops the output results by the confidence. Each word detected is associated with a confidence value from 0-100, the greater the value the more likely the word is to be accurate. By setting this to a integer value all words with a confidence beneath that will be removed from the results. A good value for this is 30 to 50 depending on the clarity of the image or the accuracy of the desired results.
## -r --min-ratio.
This specifies the minimum axis ratio desired to keep a result. Each word detected has a bounding box placed around where it thinks the word is. The program has a tendency to detect "words" which are extremely vertically stretched. To prevent this a ratio is calculated for each word where `ratio = width/height`. Then if the value of ratio is is less than the defined `-r` value, the word will be omitted from the final results. When wanting to detect only words and not single characters this feature works well left at around 2. If wanting to additionally detect single characters this value should be decreased. **This parameters has a default value of one**.
## -mw --min-width
The minimum width parameter specifies the minimum width (in pixels) which the bounding box needs to be to be included in the final output. This prevents the detection of incorrect absurdly small or thin bounding boxes
## -mh --min-height
This parameter is the exact same as the `--min-width` parameter, except applied to the height of the bounding box.
## -v --view-type
This parameter is outdated and may not work as expected.  
There are two possible inputs:  
* `"n"` - Normal viewing (can also be enabled by not calling this parameter
* `"i"` - Inverted viewing (inverts the display image)
## -oh --output-height
This is a *suprisingly important parameter*. This defines the output height of the final image when it is displayed. It has a default parameter of `720`. If the output height is greater than the vertical resolution of the monitor being used then the output will be cropped to the size of the screen, meaning that detected words will be lost (they will still be exported, just not be shown in the output image due to them being cropped out). Additionally this parameter maintains the aspect ratio of the image, meaning that as output height decreases, output width decreases as well. This could mean that output height could be set correctly, but if the image was too wide it still may go off the screen.
## -oi --override-invert
If the median color of the grayscale version of the image is less than 128 out of a possible 255 the image is designated as "dark". The image will then normally be inverted before other filters and scanning is applied. This setting overrides this operation.
## -ir --result-consolidation
The majority of words detected will be detected multiple times. This is due to segmentation overlap as well as words being able to be detected at a multitude of angles. To remove duplicate detections bounding boxes which overlap by a percentage which is greater than the one defined by `-ir` will be merged into one. By default this value is equal to 0.05 (5%). To enable/disable this feature the `-rc` setting can be called.

---
# During the program
During execution of the program multiple things will be printed to terminal, explaining the status of the operation. These will be output in the following order:
* Folder Name
* Image Name
* This image requires inversion. (If not this message is not displayed)
* Application of high pass filter
* Application of contrast filter
* Detecting lines
* Merging lines
* Segmenting the image
* Scanning segments
* Displaying image/output data
* Time to run program

---
# Output/Understanding the Results
In the same folder as the image being scanned the data is output. It is always returned in a file called `out.csv`.  
out.csv has the following format:
|Word|Point1|Point2|Point3|Point4|Confidence|
|---|---|---|---|---|---|
|WORD1|(123,456)|(010,101)|(202,020)|(654,321)|67|
|Word2|(345,1345)|(342,5646)|(324,2356)|(286,6543)|87|
|word3|(34,12)|(48,26)|(97,46)|(46,87)|45|

In the file this looks like
```
WORD1,(123,456),(010,101),(202,020),(654,321),67
Word2,(345,1345),(342,5646),(324,2356),(286,6543),87
word3,(34,12),(48,26),(97,46),(46,87),45
```
Points 1 through 4 form the 4 corners of a rectangular bounding box around the detected word. This bounding box will be tilted in the angle which the word was detected at. Additionally if result consolidation is enabled this bounding box is the largest box which the word was detected in.
