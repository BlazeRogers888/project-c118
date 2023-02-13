import cv2

body_cascade = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Create our body classifier


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale

    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # Pass frame to our body classifier

    bodies = body_cascade.detectMultiScale(grey)

    for (x,y,w,h) in bodies:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("Web cam", frame)
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
