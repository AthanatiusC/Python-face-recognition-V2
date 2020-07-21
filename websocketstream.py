from flask import Flask, render_template,request
from flask_socketio import SocketIO, send, emit,Namespace
from threading import Thread
import cv2
from base64 import b64encode

class Streamer:
    def __init__(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self.encode = {}
        self.clients = []

    def StartStreaming(self):
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

        @self.socketio.on('message')
        def handle_message(message):
            send(message)
            
        @self.socketio.on('stream')
        def handle_my_custom_event(cam_id):
            while request.sid in self.clients:
                if len(self.encode) > 0:
                    # data = []
                    try:
                        frame = self.encode[cam_id]
                        width = int(frame.shape[1] * 40 / 100)
                        height = int(frame.shape[0] * 40 / 100)
                        retval, buffer = cv2.imencode('.jpg', cv2.resize(frame,(width,height)))
                        jpg_as_text = b64encode(buffer)
                        emit('serverstream{}'.format(cam_id), jpg_as_text.decode("utf-8"))
                        # cameras = list(self.encode.items())[int(cam_id)]
                    except:
                        emit('message', 'Camera id doesnt exist')
                        return 
                    self.socketio.sleep(0.01)
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
    # vid1 = cv2.VideoCapture("rtsp://192.168.1.103:8089/h264_pcm.sdp")
    while True:
        ret,frame = vid.read()
        # ret,frame1 = vid1.read()
        # stream.test("test")
        stream.encode.update({"0":frame})
        # stream.encode.update({"1": frame1})
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            vid.release()
            cv2.destroyAllWindows()    
main()
stream.stop()
