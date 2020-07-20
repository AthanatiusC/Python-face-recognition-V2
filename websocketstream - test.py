from flask import Flask, render_template,request
from flask_socketio import SocketIO, send, emit
from threading import Thread
import base64
import cv2



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
                    data = []
                    try:
                        cameras = list(self.encode.items())[int(cam_id)]
                    except:
                        emit('message', 'Camera id doesnt exist')
                        return  
                    for cam in cameras:
                        data.append(cam)
                    emit('serverstream', data[1])
                    data.clear()
                    self.socketio.sleep(0.01)

        self.thread = Thread(
            target=self.socketio.run,
            args=(self.app,"0.0.0.0","9090",))
        self.thread.start()

    def stop(self):
        self.thread.join()

stream = Streamer()
stream.StartStreaming()
# def run():
#     socketio.run(app,host="0.0.0.0",port="8080")

def main():
    vid = cv2.VideoCapture(0)
    while True:
        ret,frame = vid.read()
        # cv2.imshow("Frame",frame)
        retval, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        stream.encode.clear()
        stream.encode.update({"frame":jpg_as_text.decode("utf-8")})
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            vid.release()
            cv2.destroyAllWindows()    
main()
stream.stop()