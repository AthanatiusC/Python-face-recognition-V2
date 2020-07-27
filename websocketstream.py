from flask import Flask, render_template,request
from flask_socketio import SocketIO, send, emit,Namespace
from threading import Thread
import cv2
from base64 import b64encode

online_cameras = {"0":0,"1":1}
datas = []
class Streamer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.encode = {}
        self.clients = []
        self.multi_cam = False

    def StartStreaming(self):
        @self.socketio.on('connect')
        def connected():
            print("User {} connected".format(request.sid))
            emit("message","Connected to server")
            emit("online_cameras",online_cameras)

            self.clients.append(request.sid)

        @self.socketio.on('disconnect')
        def disconnect():
            print("{} Disconnected".format(request.sid))
            emit("message","Disconnected from server")
            self.clients.remove(request.sid)

        @self.socketio.on('message')
        def handle_message(message):
            emit('message',message)
        
        @self.socketio.on('multi-stream-stop')
        def stopAllStream():
            print("Stop")
            self.multi_cam = False
        
        @self.socketio.on('multi-stream')
        def getAll():
            self.multi_cam = True
            while self.multi_cam == True:
                if not request.sid in self.clients:
                    break
                if len(self.encode) > 0:                   
                    temp ={}
                    for key in self.encode:
                        frame = self.encode[key]
                        width = int(frame.shape[1] * 20 / 100)
                        height = int(frame.shape[0] * 20 / 100)
                        retval, buffer = cv2.imencode('.jpg', cv2.resize(frame,(width,height)))
                        jpg_as_text = b64encode(buffer)
                        temp.update({key:jpg_as_text.decode("utf-8")})
                    emit('multi-stream-recieve', temp)
                    self.socketio.sleep(0.01)
                else:
                    break
            print("Multi stream closed")
            return

        @self.socketio.on('stream')
        def getOne(cam_id):
            while True:
                if not request.sid in self.clients:
                    break
                if len(datas) >0:
                    emit("result",datas.pop())
                if len(self.encode) > 0:
                    try:
                        frame = self.encode[str(cam_id)]
                        width = int(frame.shape[1] * 50 / 100)
                        height = int(frame.shape[0] * 50 / 100)
                        retval, buffer = cv2.imencode('.jpg', cv2.resize(frame,(width,height)))
                        jpg_as_text = b64encode(buffer)
                        emit('serverstream{}'.format(cam_id), jpg_as_text.decode("utf-8"))
                        # cameras = list(self.encode.items())[int(cam_id)]
                    except Exception as e:
                        emit('message', 'Camera id doesnt exist')
                        break
                    self.socketio.sleep(0.01)
                else:
                    break
            return

        self.thread = Thread(
            target=self.socketio.run,
            args=(self.app,"0.0.0.0","8080",))
        self.thread.start()

    def stop(self):
        self.thread.join()

stream = Streamer()
stream.StartStreaming()

def main():
    vid = cv2.VideoCapture(0)
    vid1 = cv2.VideoCapture("rtsp://192.168.1.103:8089/h264_pcm.sdp")
    datas.append({"message":"test","data":{"Name":"Test","Accu":"90%"}})
    while True:
        ret,frame = vid.read()
        ret,frame1 = vid1.read()
        # stream.test("test")
        stream.encode.update({"0":frame})
        stream.encode.update({"1": frame1})
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            vid.release()
            cv2.destroyAllWindows()    
main()
stream.stop()
