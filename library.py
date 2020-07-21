import os,requests, time, pafy, cv2,json
from flask import Flask, render_template,request
from flask_socketio import SocketIO, send, emit
import numpy as np
from align import AlignDlib
from base64 import b64encode
from face_detect import MTCNN
from threading import Thread
from multiprocessing import Process,Queue
from queue import Queue
import argparse
import sys

# import websocketstream
datas = []
class InitDetectionModel:
    def __init__(self, long_dist):
        #VARIABLE DECLARATION
        self.model = None
        self.model = self.InitModel(long_dist)

    def InitModel(self,long_dist):
        if long_dist:
            model = {'name':'mtcnn','model':self.load_mtcnn()}
        else:
            model = {'name':'resnet','model':self.load_resnet()}
        return model
        
            
    def load_resnet(self):
        resnet = cv2.dnn.readNetFromCaffe("model/ResNet/deploy.prototxt","model/ResNet/res10_300x300_ssd_iter_140000.caffemodel")
        resnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        resnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return resnet

    def load_mtcnn(self):
        pnet_model_path = './model/MTCNN/pnet'
        rnet_model_path = './model/MTCNN/rnet'
        onet_model_path = './model/MTCNN/onet'
        mtcnn = MTCNN(pnet_model_path,
                        rnet_model_path,
                        onet_model_path)
        return mtcnn

class Config:
    def __init__(self):
        self.configuration = None
    
    def ReadConfiguration(self):
        with open('configuration.json') as json_file:
            self.configuration = json.load(json_file)
        return self.configuration

class Api:
    def __init__(self,socket):
        self.data_queue = Queue(maxsize=20)
        self.stopped = False
        self.socketio = socket
        self.clients = []
        self.datas = Queue()

    def add_data(self, data):
        # print("adding.. total {}".format(self.data_queue.qsize()))
        if self.data_queue.not_full:
            self.data_queue.put(data)
        data = None
        return

    def start_sender(self):
        self.thread = Thread(target=self.send_data)
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join()

    def send_data(self):
        while not self.stopped:
            if self.data_queue.not_empty:
                data = self.data_queue.get()
                face = data["face"]
                vector = data["vector"]
                address = data["address"]
                camera_id = data["camera_id"]
                try:
                    retval,buffer = cv2.imencode(".jpg",face )
                    string_bytes = b64encode(buffer)
                    data = {"image": string_bytes.decode('utf-8'), "camera_id": camera_id, "embeddings": np.array(vector[0]).astype("float64").tolist()}
                    res = requests.post(url="http://{}:8088/api/v2/user/recognize".format(address), json=data)
                    # datas.put(res.json())
                    datas.append(res.json())
                    # if res != None:
                    #     self.socketio.emit("result", res.json())
                    # print(res.json())
                except Exception as e:
                    pass

class Stream:
    def __init__(self,online_camera):
        self.app = Flask("Face Recognition")
        
        self.camera_list = online_camera
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.clients = []
        self.frames = {}
        self.is_streaming = True
    
    def StartStreaming(self):
        self.thread = Thread(
            target=self.socketio.run,
            args=(self.app,"0.0.0.0","8080",))
        self.thread.start()
        print("\n\n Stream socket initiated \n\n")
        return self.socketio

    def handle(self):
        @self.socketio.on('connect')
        def connected():
            print("User {} connected".format(request.sid))
            emit("message","Connected to server")
            self.clients.append(request.sid)

        @self.socketio.on('disconnect')
        def disconnect():
            print("{} Disconnected".format(request.sid))
            emit("message","Disconnected from server")
            self.clients.remove(request.sid)

        @self.socketio.on('online_camera')
        def handle_message(message):
            send(self.camera_list)

        @self.socketio.on('stream')
        def upstream(cam_id):
            while request.sid in self.clients:
                if len(datas) >0:
                    emit("result",datas.pop())
                if len(self.frames) > 0:
                    # data = []
                    try:
                        frame = self.frames[cam_id]
                        width = int(frame.shape[1] * 40 / 100)
                        height = int(frame.shape[0] * 40 / 100)
                        retval, buffer = cv2.imencode('.jpg', cv2.resize(frame,(width,height)))
                        jpg_as_text = b64encode(buffer)
                        emit('serverstream{}'.format(cam_id), jpg_as_text.decode("utf-8"))
                        # if detection_result.not_empty:
                        #     emit("result", detection_result.get(), broadcast=True)
                    except Exception as e:
                        emit('message', 'Camera id doesnt exist')
                        print(e)
                        return 
                    self.socketio.sleep(0.01)
            return

    def stop(self):
        self.thread.join()

class Recognition:
    def __init__(self):
        self.load_model()
        pass

    def load_model(self):
        self.embedder = cv2.dnn.readNetFromTorch("model/Openface/openface_nn4.small2.v1.t7")
        self.embedder.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.embedder.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.align_tools = AlignDlib(os.path.join("model","landmarks.dat"))
        
    def align_face(self, frame):
        return self.align_tools.align(96, frame, self.align_tools.getLargestFaceBoundingBox(frame), 
                            landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    
    def extract_embeddings(self, aligned_face):
        try:
            faceBlob = cv2.dnn.blobFromImage(aligned_face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        except:
            return
        self.embedder.setInput(faceBlob)
        vector = self.embedder.forward()
        return vector

class FaceDetector:
    def __init__(self,Model):
        self.model = Model["model"]
        self.model_name = Model["name"]

    def DetectFace(self, frame):
        result = None
        if self.model_name == "resnet":
            result = self.resnet_mode(frame)
        elif self.model_name == "mtcnn":
            result = self.mtcnn_mode(frame)
        return result

    def resnet_mode(self, frame):
        faces = []
        boxes = []
        (h, w) = frame.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
        self.model.setInput(imageBlob)
        detections = self.model.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = frame[startY:endY, startX:endX]
                # (fH, fW) = face.shape[:2]
                # if fW < 20 or fH < 20:
                #     continue
                boxes.append(box)
                faces.append(face)
                y = startY - 10 if startY - 10 > 10 else startY + 10
        # print("resnet : {}".format(len(faces)))
        return {"face":faces,"boxes":boxes}

    def mtcnn_mode(self, frame):
        (h, w) = frame.shape[:2]
        faces = []
        bounding_boxes, landmarks = self.model.detect(
            img=frame, min_size=50, factor=0.709,
            score_threshold=[0.8, 0.8, 0.8]
        )
        for idx, bbox in enumerate(bounding_boxes):
            h0 = max(int(round(bbox[0])), 0)
            w0 = max(int(round(bbox[1])), 0)
            h1 = min(int(round(bbox[2])), h - 1)
            w1 = min(int(round(bbox[3])), w - 1)
            faces.append(frame[h0 - 30:h1 + 30, w0 - 30:w1 + 30])
            # rectangle(frame, (w0, h   0), (w1, h1), (255, 255, 255), 1)
        return {"face":faces,"boxes":bounding_boxes}

    def draw_overlay(self, frame, boxes, frame_rate):
        for box in boxes:
            try:
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            except:
                (h, w) = frame.shape[:2]
                for idx, bbox in enumerate(boxes):
                    h0 = max(int(round(bbox[0])), 0)
                    w0 = max(int(round(bbox[1])), 0)
                    h1 = min(int(round(bbox[2])), h - 1)
                    w1 = min(int(round(bbox[3])), w - 1)
                    cv2.rectangle(frame, (w0, h0), (w1, h1), (255, 255, 255), 1)
        if frame_rate < 20:
            cv2.putText(frame, "Fps : {}".format(str(frame_rate)), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame,"Fps : {}".format(str(frame_rate)),(20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


class FacialRecognition:
    def __init__(self):
        self.not_finished = True
        parser = argparse.ArgumentParser(description='Input the Application Config.')
        parser.add_argument('--long_distance',default=False,type=bool,help='Configure if camera should run on resnet or MTCNN')
        parser.add_argument('--streaming_fps',default=30,type=int,help='FPS Streaming limit')
        self.args = vars(parser.parse_args())

    def Start(self):
        config = Config()
        configuration = config.ReadConfiguration()
        self.thread_list = []
        self.online_camera =[]
        stream = Stream(self.online_camera)
        socket = stream.StartStreaming()
        stream.handle()
        sender = Api(socket)
        # sender.handlesocket()
        for config in configuration:
            print("Checking {}".format(config["name"]))
            if not self.check_video(config["address"]):
                continue
            self.online_camera.append(config["name"])
            thread = Thread(target=self.main,args=(config,sender,stream))
            thread.setName(config["name"])
            self.thread_list.append(thread)
        for thread in self.thread_list:
            thread.start()
        for thread in self.thread_list:
            thread.join()
        print("program finished")
            
    def main(self, config,sender,stream):
        model = InitDetectionModel(self.args["long_distance"])
        detector = FaceDetector(model.model)
        recognition = Recognition()
        frame_interval = config["recognition_per_second"]  # Number of frames after which to run face detection
        fps_display_interval = config["fps_update_interval_second"]  # seconds
        frame_rate = 0
        frame_count = 0
        start_time = time.time()
        sender.start_sender()
        camera = config["address"]
        camera_id = config["id"]
        boxes=None
        # (startX, startY, endX, endY) = None
        while self.not_finished:
            try:
                vid = cv2.VideoCapture(int(camera))
            except:
                vid = cv2.VideoCapture(camera)
            while self.not_finished:
                ret, frame = vid.read()
                if not ret:
                    print("Frame get failed")
                    break
                if (frame_count % frame_interval) == 0:
                    det = detector.DetectFace(frame)
                    faces = det["face"]
                    boxes = det["boxes"]
                    for face in faces:
                        aligned_face = recognition.align_face(face)
                        vector = recognition.extract_embeddings(aligned_face)
                        start = time.time()        
                        if type(vector) != None:
                            # if sender.data_queue.not_full:
                            data = {"face":face,"vector":vector,"camera_id":camera_id,"address":config["server_ip"]}
                            #     sender.data_queue.put(data)
                            t = Thread(target=sender.add_data,args=(data,))
                            t.daemon = True
                            t.start()
                        # print("thread {} : {}".format(config["name"],time.time()-start))
                    
                    end_time = time.time()
                    if (end_time - start_time) > fps_display_interval:
                        frame_rate = int(frame_count / (end_time - start_time))
                        start_time = time.time()
                        frame_count = 0
                
                try:
                    detector.draw_overlay(frame,boxes,frame_rate)
                except Exception as e:
                    print(e)
                frame_count += 1
                stream.frames.update({config["name"]: frame})
                # print(len(stream.frames))
                # cv2.imshow(config["name"],frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    cv2.destroyAllWindows()
                    self.not_finished = False
                    os._exit(0)
        self.not_finished = False
        stream.stop_streaming()
        sender.stop()
    
    def check_video(self, address):
        try:
            vid = cv2.VideoCapture(int(address))
        except:
            vid = cv2.VideoCapture(address)
        ret, frame = vid.read()
        vid.release()
        if ret:
            return True
        else:
            return False