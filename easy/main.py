import cv2
import imutils
import numpy as np
import easyocr
import pandas as pd 
import datetime

capture = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'])

while capture.isOpened():
    ref, frame = capture.read()
    if ref:
   
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(grey, 11, 17, 17)
        canny = cv2.Canny(blur, 30, 200)

     
        key_points = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(key_points)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        for cont in contours:
            epsilon = 0.02 * cv2.arcLength(cont, True)
            pol = cv2.approxPolyDP(cont, epsilon, True)
            if len(pol) == 4:
                location = pol
                break  

        if location is not None:
            mask = np.zeros(grey.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0, 255, -1)

        
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))

           
            height, width = grey.shape
            x1, x2 = max(0, x1), min(height - 1, x2)
            y1, y2 = max(0, y1), min(width - 1, y2)

            img = grey[x1:x2+1, y1:y2+1]

         
            result = reader.readtext(img)
            if result:
                print(result)
            else:
                print("No text detected.")

          
            cv2.rectangle(frame, (y1, x1), (y2, x2), (255, 0, 0), 2)

     
        cv2.imshow("Window", frame)

     
        if cv2.waitKey(24) & 0xFF == ord("q"):
            break
    else:
        break


capture.release()
cv2.destroyAllWindows()
