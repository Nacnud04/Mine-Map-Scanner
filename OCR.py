import cv2
from pytesseract import *
pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# open up dictionary, and store in memory
fd = open('words_alpha.txt',"r")    # open the file in read mode
file_contents = fd.readlines()

# check if word fits in dictionary
def check_if_word(word):
    word = f'{word}\n'
    status = False
    if(word in file_contents):  # check if word is present or not
        status = True
    else:
        status = False 
    return status

# main ocr function
def OCR(img, theta = float, minheight=int, minwidth=int, minconf=int, minratio=int, location=tuple, xoffset=None, yoffset=None):
    # detect
    results = pytesseract.image_to_data(img, output_type=Output.DICT)

    data = []

    # iterate over each of the text localizations
    for i in range(0, len(results["text"])):
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        if h != 0:
            ratio = w/h

        # We will also extract the OCR text itself along
        # with the confidence of the text localization
        text = results["text"][i]
        conf = int(float(results["conf"][i]))

        show = True



        ### BEGIN TO FILTER CONTENTS

        # reasons to remove
        # remove blank detections
        texthex = ':'.join(hex(ord(x))[2:] for x in text)
        if texthex == "" or texthex == "20" or texthex == "20:20" or texthex == "20:20:20" or texthex == "20:20:20:20" or texthex == "20:20:20:20:20" or ratio <= minratio:
            show = False
        # check dictionary
        checkstatus = check_if_word(text.lower())
        if checkstatus == False and show != False: # and ignore != "d":
            show = False
        # check string length
        if len(text) <= 2:
            show = False
        # test width and height
        if minwidth >= w or minheight >= h:
            show = False

        additionalrestrictions = False
        if additionalrestrictions == True and text != text.upper():
            show = False

        # exceptions
        if text.lower() == "no" or text.lower() == "no.":
            show = True
        try:
            integer = int(text.strip("."))
            if integer >= 10:
                show = True
        except:
            pass

        if conf > minconf and show == True and ratio >= minratio:
            data.append((text, conf, location, theta, x, y, w, h, xoffset, yoffset))
            #print(f'Confidence: {conf}  |  Text: {text}  |  Hex: {texthex}')
    return data