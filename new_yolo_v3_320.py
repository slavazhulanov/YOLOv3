import cv2
import numpy as np

net = cv2.dnn.readNet('model_data/yolov3.weights', 'model_data/yolov3.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
classes = open('model_data/coco.names').read().strip().split('\n')

cap = cv2.VideoCapture(0)
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter('video_out/10_sec_out_v32.avi',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))


while(cap.isOpened()):
    ret, img = cap.read()
    H, W = img.shape[:2]
    conf = 0.5

    blop = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blop)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf:
                center_x = int(detection[0] * W)
                center_y = int(detection[1] * H)
                w = int(detection[2] * W)
                h = int(detection[3] * H)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf - 0.1)
    font = cv2.FONT_HERSHEY_PLAIN
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = [int(c) for c in colors[class_ids[i]]]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y - 5), font, 1, color, 2)

    if ret == True:
        out.write(img)
    else:
        break