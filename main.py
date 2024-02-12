import cv2
import os
import numpy as np
import dlib
import imutils
from imutils import face_utils
from tensorflow.keras.preprocessing import image
from statistics import mode
from keras.models import load_model
from gtts import gTTS
from pygame import mixer

model = load_model('C:\\Users\\ASUS\\OneDrive\\Desktop\\Lip_test_final\\Lip_test_final\\trained_model.h5')

def direct(val):
    dir = val
    location = 'static/'
    path = os.path.join(location, dir)
    if os.path.exists(path):
        os.remove(path) 
    os.mkdir(path)

def toFrames(vid, saved_path):
    vidcap = cv2.VideoCapture(vid)
    success, image = vidcap.read()
    count = 0
    success = True

    while success:
        success, image = vidcap.read()
        if success:
            print('Read a new frame: ',count+1)
            cv2.imwrite(saved_path + '/frame_%d.jpg' % count, image)
            count += 1

    if not success:
        count -= 1
        os.remove(saved_path + '/frame_%d.jpg' % count)

def mouth_detection(shape_predictor, img, saved_name, saved_path):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name == 'mouth':

                clone = image.copy()
                cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)

                for (x, y) in shape[i:j]:

                    cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    roi = image[y-12:y + h + 12, x-5:x + w + 5]
                    roi = imutils.resize(roi, width=125, inter=cv2.INTER_CUBIC)
                    cv2.imwrite(os.path.join(saved_path, saved_name), roi)

classes = ['Hi', 'How Are You',  'Thank You']

def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(227, 227))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    result.append(str(classes[index]))

def speech(text):
    mytext = text
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("result4.mp3")

if __name__ == "__main__":

    # direct('img')
    video = 'static/1.mp4'
    saved_dir = 'static/img/'
    print('Video to Frame -> Start')
    toFrames(video, saved_dir)
    print('Video to Frame -> Completed')

    # direct('frames')
    files = os.listdir(saved_dir)
    frame_dir = "static/frames/"
    print('Frame to Lip -> Start')
    for file in files:
        file_path = saved_dir + file
        mouth_detection('shape_predictor_68_face_landmarks (1).dat', file_path, file, frame_dir)
        print('Processed ' + file)
    print('Frame to lip -> Completed')

    path = frame_dir
    result = []
    print('Predicted....')
    for i in os.listdir(path):
        predict_image(path+i, model)

    temp = [wrd for sub in result for wrd in sub.split()]
    res = mode(temp)
    print(res)
    speech(res)
    mixer.init()
    mixer.music.load('result4.mp3')
    mixer.music.play()
    print('Completed')

