# Crea un script que llame a la camara web y abra yolo4 sobre ella
# Existe un folder llamado yolo-coco donde se encuentran ("./yolo-coco/coco.names" "./yolo-coco/yolov4-p6.cfg" "./yolo-coco/yolov4-p6.weights")

# importing the necessary packages
import numpy as np
import time
import cv2
from cv2 import cuda

print(cuda.printCudaDeviceInfo(0))
# load the COCO class labels our YOLO model was trained on
labelsPath = "./yolo-coco/coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = "./yolo-coco/yolov4-tiny.weights"
# yolov4-p6 is too heavy...I need better graphic card :(
configPath = "./yolo-coco/yolov4-tiny.cfg"

# load our YOLO object detector trained on COCO dataset (80 classes)
print(" [INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# load our camera image and grab its spatial dimensions
#cap = cv2.VideoCapture(0)


def get_layerOutputs(frame):
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    return layerOutputs


def get_info(layeroutputs, img_width, img_height):
    boxes = []
    confidences = []
    class_ids = []
    # loop over each of the layer outputs
    for output in layeroutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idsx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    return boxes, confidences, class_ids, idsx


def draw_boxes(img, boxes, confidences, class_ids, idsx, draw_person=False):
    flag = False
    # ensure at least one detection exists
    if len(boxes) > 0:
        # loop over the indexes we are keeping
        for i in idsx.flatten():
            if not LABELS[class_ids[i]] == "human":
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[class_ids[i]]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                flag = True
                if draw_person:
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    # draw a bounding box rectangle and label on the image
                    color = [int(c) for c in COLORS[class_ids[i]]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return flag


'''''
# loop over frames from the video file stream
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # load our input image and grab its spatial dimensions
    (H, W) = frame.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    layerOutputs = get_layerOutputs(net)
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes, confidences, classIDs, idxs = get_info(layerOutputs, W, H)
    # draw the final bounding boxes and labels on the output frame
    draw_boxes(frame, boxes, confidences, classIDs, idxs)
    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()
'''''
