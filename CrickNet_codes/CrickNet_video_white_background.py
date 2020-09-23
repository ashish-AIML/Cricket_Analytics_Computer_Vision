# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import math

def weird_division(n, d):
    return abs(n / d) if d else 0

def weird_division1(n, d):
    return n / d if d else 0

ball_count=1
frame_count=1

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-l", "--labels", required=True,
	help="base path to labels file")
ap.add_argument("-w", "--weight", required=True,
	help="base path to weights file")
ap.add_argument("-y", "--cfg", required=True,
	help="base path to cfg file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

###########################################################################################
##create white frame
def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color

    return image



###################################################################################



# load the COCO class labels our YOLO model was trained on
labelsPath = args["labels"]
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = args["weight"]
configPath = args["cfg"]


# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
print(weightsPath)

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
    
 ##############################################################
 
	width, height = 720, 1280
	white = (255, 255, 255)
	frame1 = create_blank(width, height, rgb_color=white)
    
################################################################   

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

###################################################################################################
#ball count, centre coordinates and angle calculations
			if i==0:
				cx=x+(w/2)
				cy=y+(h/2)
				print('frame: ', "%.2d"%frame_count, 'Ball centre x: ', "%.2f"%cx, 'centre y: ', "%.2f"%cy )
#				text2="cx:"+str(cx)
#				text3="cy:"+str(cy)            
#				cv2.putText(frame, text2, (200,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
#				cv2.putText(frame, text3, (400,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 3)
				if ball_count==1:
					cx1=cx
					cy1=cy
				elif ball_count==2:
					cx2=cx
					cy2=cy
				else:
					if ball_count%4==0:     
						m1=weird_division1((cy2-cy1), (cx2-cx1))
						m2=weird_division1((cy-cy2), (cx-cx2))
						angle=math.degrees(math.atan(weird_division((m2-m1),(1+(m1*m2)))))
						print('m1: ', "%.2f"%m1, 'm2:', "%.2f"%m2,'angle:', "%.2f"%angle)
						dot=math.sqrt(((cy2-cy1)*(cy2-cy1))+((cx2-cx1)*(cx2-cx1)))*math.sqrt(((cy-cy2)*(cy-cy2))+((cx-cx2)*(cx-cx2)))*math.cos(angle)
						print('Dot Product: ', "%.2f"%dot)
#						text4="Ang:"+str(int(angle))
#						cv2.putText(frame, text4, (0,400),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
						cx1=cx2
						cy1=cy2
						cx2=cx
						cy2=cy			                
				ball_count+=1

			if i==2:
				bcx=x+(w/2)
				bcy=y+(h/2)
				dist=math.sqrt(((bcx-cx)*(bcx-cx))+((bcy-cy)*(bcy-cy)))
				print('frame: ', "%.2d"%frame_count,'Bat centre x: ', "%.2f"%bcx, 'centre y: ', "%.2f"%bcy,'Distance btw bat and ball: ', "%.2f"%dist )

			if i==1:
				pcx=x+(w/2)
				pcy=y+(h/2)
				print('frame: ', "%.2d"%frame_count,'Pitch centre x: ', "%.2f"%pcx,  'centre y: ', "%.2f"%pcy)
                
			if i==3:
				bacx=x+(w/2)
				bacy=y+(h/2)
				print('frame: ', "%.2d"%frame_count,'Batsman centre x: ', "%.2f"%bacx, 'centre y: ', "%.2f"%bacy)


################################################################################################### 

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame1, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame1, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
################################################################################################### 
#printing fr,ae number
	text1=str(frame_count)
	cv2.putText(frame1, text1, (0,100),
				cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
	frame_count=frame_count+1
####################################################################################################

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame1)
    

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()