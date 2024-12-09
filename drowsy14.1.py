
#!/usr/bin/env python
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords
import datetime
from flask import Flask, Response, render_template_string
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
LED_PIN = 17
BUZZER_PIN = 23
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Initialize the LCD with I2C address
lcd = CharLCD('PCF8574', 0x27)

# initialize dlib's face detector and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# initialize the video stream and sleep for a bit
print("[INFO] initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
if vs.stream is None or not vs.stream.isOpened():
    print("[ERROR] Unable to open camera")
    vs.stop()
    GPIO.cleanup()
    lcd.clear()
    exit()

# Set the desired frame width and height
frame_width = 640
frame_height = 360

# Flask app setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string('''
        <html>
            <head>
                <title>Drowsiness Detection</title>
            </head>
            <body>
                <h1>Drowsiness Detection Camera Feed</h1>
                <img src="{{ url_for('video_feed') }}" width="640" height="360">
            </body>
        </html>
    ''')

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0
ALARM_ON = False
ALARM_START = None

def generate():
    global COUNTER, ALARM_ON, ALARM_START
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=frame_width, height=frame_height)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        rects = detector(gray, 0)

        for rect in rects:
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
            (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                if not ALARM_ON:
                    lcd.cursor_pos = (0, 4)
                    lcd.write_string('All Clear!')
                    
                    ALARM_START = datetime.datetime.now()
                    ALARM_ON = True
                elapsed_time = (datetime.datetime.now() - ALARM_START).total_seconds()
                if elapsed_time >= 3:
                    GPIO.output(LED_PIN, GPIO.HIGH)
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)
                    lcd.clear()
                    lcd.cursor_pos = (0, 3)
                    lcd.write_string('Take a Break!')
                    time.sleep(3)
                    GPIO.output(LED_PIN, GPIO.LOW)
                    GPIO.output(BUZZER_PIN, GPIO.LOW)
                    lcd.clear()
                    ALARM_ON = False
                    ALARM_START = None
            else:
                ALARM_ON = False
                ALARM_START = None

            (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
            mouth = shape[mStart:mEnd]
            mouthMAR = mouth_aspect_ratio(mouth)
            mar = mouthMAR
            mouthHull = cv2.convexHull(mouth)
            # Commenting out the mouth line
            # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            #cv2.putText(frame, "MAR: {:.2f}".format(mar), (450, 20), 
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Yawning!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            image_points = np.zeros((6, 2), dtype='double')
            for (i, (x, y)) in enumerate(shape):
                if i == 33:
                    #image_points[0] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, -1)
                    # Commenting out the numbers
                    # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                    #image_points[1] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, -1)
                    # Commenting out the numbers
                    # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:
                    #image_points[2] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, -1)
                    # Commenting out the numbers
                    # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:
                    #image_points[3] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, -1)
                    # Commenting out the numbers
                    # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                    #image_points[4] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, -1)
                    # Commenting out the numbers
                    # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 54:
                    #image_points[5] = np.array([x, y], dtype='double')
                    cv2.circle(frame, (x, y), 1, -1)
                    # Commenting out the numbers
                    # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                # Commenting out the red dots and lines
                # else:
                #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                #     cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            (head_tilt_degree, start_point, end_point, 
                end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

            # Commenting out the lines
            # cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            # cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

            # Commenting out the head tilt degree text
            # if head_tilt_degree:
            #     cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        vs.stop()
        GPIO.cleanup()
        lcd.clear()
