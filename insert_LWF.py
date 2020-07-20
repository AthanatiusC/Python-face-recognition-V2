import os, cv2, pymongo, bcrypt, random
import numpy as np
from align import AlignDlib
import library as Lib
import request

conn = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = conn["smart_school"]
mycol = mydb["users"]

Recognition = Lib.Recognition()
model = Lib.InitDetectionModel(False)
Detection = Lib.FaceDetector(model.model)

dirs = "D:/lfw/100/"
names = os.listdir(dirs)
identity = []

for name in names:
    vectors = []
    identity_path = dirs + name
    image_list = os.listdir(identity_path)
    for filename in image_list:
        img_paths = identity_path+"/"+filename
        print("Processing : {}".format(img_paths))
        img = cv2.imread(img_paths)

        det = Detection.DetectFace(img)
        faces = det["face"]
        for face in faces:
            aligned_face = Recognition.align_face(face)
            try:
                cv2.imshow("img", aligned_face)
            #     cv2.waitKey(0)
            except:
                continue
            vector = Recognition.extract_embeddings(aligned_face)
            vectors.append(np.array(vector[0]).astype("float64").tolist())
    user_id = ""
    for x in range(3):
        user_id+=str(random.randint(0,9))
    identity.append({
        "name": "Testing "+name,
        "username": name,
        "password": bcrypt.hashpw(b"user", bcrypt.gensalt()),
        "role": "Student",
        "phone":"Testing",
        "address": "Testing",
        "class_id":"5d540ba516171e39480051b0",
        "user_id":user_id,
        "embeddings": vectors})
# print(identity)

# x = mycol.insert_many(identity)
# print(x.inserted_ids)

# print(identity)

# print(identity[2]["embeddings"][0])

# data = {"image": string_bytes.decode('utf-8'), "camera_id": camera_id, "embeddings": np.array(vector[0]).astype("float64").tolist()}
# res = requests.post(url="http://{}:8088/api/v2/user/recognize".format(address), json=data)