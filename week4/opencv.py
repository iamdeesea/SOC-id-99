import cv2 as cv

img = cv.imread('photo.jpg')
# cv.imshow('DC',img)

#converting to grayscale
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('grey',gray)

#blur
# blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
# cv.imshow('blur',blur)

#edge cascade
# canny = cv.Canny(img,125,175)
# cv.imshow('Canny edited',canny)


#dilating the image
# dilated = cv.dilate(canny,(3,3),iterations=1)
# cv.imshow('dilated',dilated)

#eroding
# eroded = cv.erode(dilated,(3,3),iterations=1)
# cv.imshow('eroded',eroded)

#resize
# resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
# cv.imshow('resized',resized)

#cropped
# cropped = img[50:200,200:300]
# cv.imshow('crop',cropped)
cv.waitKey(0)
import cv2 as cv
import numpy as np
blank = np.zeros((400,400),dtype='uint8')

rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)
cv.imshow('rec',rectangle)
cv.imshow('cir',circle)

#bitwise And
bitwise_and = cv.bitwise_and(rectangle,circle)
cv.imshow('and',bitwise_and)

#bitwise or
bitwise_or = cv.bitwise_or(rectangle,circle)
cv.imshow('or',bitwise_or)

#bitwise xor
bitwise_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow('xor',bitwise_xor)

#bitwise not
bitwise_not = cv.bitwise_not(circle)
cv.imshow('not',bitwise_not)
cv.waitKey(0)
import cv2 as cv
import numpy as np
img = cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)
# cv.imshow('original',resized)
blank = np.zeros(resized.shape,dtype='uint8')
cv.imshow('Blank',blank)

gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
cv.imshow('Gray',gray)

canny = cv.Canny(resized,125,175)                         
# cv.imshow('edged',canny)
ret,thresh = cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow('thres',thresh)


contours, hierarchies = cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)}')
cv.drawContours(blank,contours,-1,(0,0,255),2)
cv.imshow('cont',blank)
cv.waitKey(0)
import cv2 as cv
import numpy as np
blank= np.zeros((500,500,3), dtype='uint8')
# img = cv.imread('catty.jpg')

# cv.imshow('Cat',img)
# cv.imshow('blank',blank)
# img= cv.imread('catty.jpg')
# cv.imshow('Cat', img)

# paint the image a certain color
# blank[200:300,300:400] = 0,255,0
# cv.imshow('green',blank)

#draw a rectangle
# cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,0,0),thickness=cv.FILLED)
# cv.imshow('Rectangle',blank)

#draw a circle
# cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),30,(0,0,255),thickness=-1)
# cv.imshow('circle',blank)

#draw a line
# cv.line(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(255,255,255),thickness=3)
# cv.imshow('line',blank)

# write text
# cv.putText(blank,'Hello',(225,255),cv.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2)
# cv.imshow('text',blank)
cv.waitKey(0)
import cv2 as cv

# Load the Haar cascade for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Start video capture
capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()

    if not isTrue:
        print("Failed to grab frame")
        break

    # Resize the frame (optional)
    resized = cv.resize(frame, (500, 500), interpolation=cv.INTER_AREA)

    # Convert to grayscale for detection
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces_rect:
        cv.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    # Show the result
    cv.imshow('Webcam Face Detection', resized)

    # Break the loop on 'd' key press
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

# Release the capture and close windows
capture.release()
cv.destroyAllWindows()
import os
import cv2 as cv
import numpy as np

p=[]
for i in os.listdir(r'C:\Users\DC\Downloads\opencv-course-master\opencv-course-master\Resources\Faces\train'):
    p.append(i)
DIR
print(p)
import cv2 as cv
import numpy as np
img= cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)

gray= cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#laplacion
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('lap',lap)

#sobel
sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)
combined_sobel=cv.bitwise_or(sobelx,sobely)
cv.imshow('Sobel X',sobelx)
cv.imshow('Sobel Y',sobely)
cv.imshow('comb',combined_sobel)

canny=cv.Canny(gray,150,175)
cv.imshow('canny',canny)
cv.waitKey(0)
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img= cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('orignal',resized)
blank = np.zeros(resized.shape[:2],dtype='uint8')

# gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray',gray)

mask = cv.circle(blank,(resized.shape[1]//2,resized.shape[0]//2),100,255,-1)
masked = cv.bitwise_and(resized,resized,mask=mask)

cv.imshow('mask',masked)
#grayscale histogram
# gray_hist = cv.calcHist([gray],[0],mask,[256],[0,256])

plt.figure()
plt.title('color Histogram')              
plt.xlabel('bims')
plt.ylabel('no. of pixels')
# plt.plot(gray_hist)
# plt.xlim(0,256)
# plt.show()

#color histogram
color = ('b','g','r')
for i,col in enumerate(color):
    hist = cv.calcHist([resized],[i],mask,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim(0,256)
    
plt.show()

# cv.waitKey(0)
import cv2 as cv
import numpy as np

img = cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)
cv.imshow('original',resized)

blank = np.zeros(resized.shape[:2],dtype='uint8')
cv.imshow('Blank Image', blank)

mask = cv.circle(blank,(resized.shape[1]//2,resized.shape[0]//2),100,255,-1)

cv.imshow('mask',mask)

masked = cv.bitwise_and(resized,resized,mask=mask)
cv.imshow('masked image',masked)

cv.waitKey(0)
import cv2 as cv    

# img = cv.imread('catty.jpg')

# cv.imshow('Cat',img)

# cv.waitKey(0)
# capture=cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     cv.imshow('Video',frame)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# capture.release()

# cv.destroyAllWindows()
# def rescaleFrame(frame,scale=0.75):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)

#     dimensions= (width,height)

#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
# cv.waitKey(0)
import cv2 as cv

# img = cv.imread('catty.jpg')
# cv.imshow('Cat',img)



# def rescaleFrame(frame, scale=0.5):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)
#     dimensions = (width,height)

#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
# def changeRes(width,height):
#     capture.set(3,width)
#     capture.set(4,height)

#reading images
# resized_image= rescaleFrame(img,scale=0.25)
# cv.imshow('cat', resized_image)
# cv.waitKey(0)
# reading videos
# capture=cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     frame_resized= rescaleFrame(frame,scale=0.2)
#     cv.imshow('Video',frame)
#     cv.imshow('Video Resized',frame_resized)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# capture.release()
# cv.waitKey(0)
import cv2 as cv

# img = cv.imread('catty.jpg')
# cv.imshow('Cat',img)



# def rescaleFrame(frame, scale=0.5):
#     width = int(frame.shape[1]*scale)
#     height = int(frame.shape[0]*scale)
#     dimensions = (width,height)

#     return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)
# def changeRes(width,height):
#     capture.set(3,width)
#     capture.set(4,height)

#reading images
# resized_image= rescaleFrame(img,scale=0.25)
# cv.imshow('cat', resized_image)
# cv.waitKey(0)
# reading videos
# capture=cv.VideoCapture(0)
# while True:
#     isTrue, frame = capture.read()
#     frame_resized= rescaleFrame(frame,scale=0.2)
#     cv.imshow('Video',frame)
#     cv.imshow('Video Resized',frame_resized)
#     if cv.waitKey(20) & 0xFF==ord('d'):
#         break

# capture.release()
# cv.waitKey(0)
import cv2 as cv

img = cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)

#averaging
average = cv.blur(resized,(7,7))
cv.imshow('avg blur',average)

#Gaussian Blur
gaussian = cv.GaussianBlur(resized,(7,7),0)
cv.imshow('gaussian',gaussian)

#median blur
median = cv.medianBlur(resized,7)
cv.imshow('median',median)

#bilateral blur
bilateral = cv.bilateralFilter(resized,10,15,15)
cv.imshow('bilateral',bilateral)
cv.waitKey(0)
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)
# cv.imshow('orignal',resized)

# plt.imshow(resized)
# plt.show()

# #bgr to grayscale
# gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

#br to hsv
hsv = cv.cvtColor(resized,cv.COLOR_BGR2HSV)
# cv.imshow('hsv',hsv)

# #bgr to l*a*b
# lab = cv.cvtColor(resized,cv.COLOR_BGR2Lab)
# cv.imshow('lab',lab)

#bgr to rgb
# rgb = cv.cvtColor(resized,cv.COLOR_BGR2RGB)
# cv.imshow('rgb',rgb)

# plt.imshow(rgb)
# plt.show()

#hsv to bgr
# hsv_bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)


cv.waitKey(0)
import cv2 as cv
import numpy as np

img = cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)
cv.imshow('org',resized)

blank = np.zeros(resized.shape[:2],dtype='uint8')
b,g,r = cv.split(resized)

blue = cv.merge([b,blank,blank])
cv.imshow('b',blue)

cv.imshow('blue',b)
cv.imshow('red',r)
cv.imshow('green',g)

print(resized.shape)
print(b.shape)
print(r.shape)
print(g.shape)

merged = cv.merge([b,g,r])
cv.imshow('merged',merged)

cv.waitKey(0)
import cv2 as cv

img = cv.imread('photo.jpg')
resized = cv.resize(img,(500,500),interpolation=1)

gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

#simple thresholding
threshold , thresh = cv.threshold(gray,100,255,cv.THRESH_BINARY)
cv.imshow('thresh',thresh)  
threshold,thresh_inv = cv.threshold(gray,100,255,cv.THRESH_BINARY_INV)
cv.imshow('threshinverse',thresh_inv)

#adaptive thresholding
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,3)
cv.imshow('adapt thresh',adaptive_thresh)

adaptive_thresh_inv = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,11,3)
cv.imshow('adapt thresh inv',adaptive_thresh_inv)
cv.waitKey(0)  
import cv2 as cv
import numpy as np
img = cv.imread('photo.jpg')


#translation
# def translation(img,x,y):
#     transMat = np.float32([[1,0,x],[0,1,y]])
#     dimensions= (img.shape[1],img.shape[0])
#     return cv.warpAffine(img,transMat,dimensions)

# translated = translation(img,100,100)
# cv.imshow('translated',translated)
# cv.waitKey(0)

#rotation
# def rotation(img,angle,rotPoint=None):
#     (height,width)= img.shape[:2]

#     if rotPoint is None:
#         rotPoint = (width//2,height//2)

#     rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
#     dimensions = (width,height)

#     return cv.warpAffine(img , rotMat,dimensions)

# rotated = rotation(img,45)
# cv.imshow('rotated',rotated)

# resizing
resized = cv.resize(img,(500,500), interpolation=cv.INTER_CUBIC)
# cv.imshow('resize',resized)
# cv.waitKey(0)

#flipping
# flip = cv.flip(resized,2)
# cv.imshow('flip',flip)


cv.waitKey(0)
