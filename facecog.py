import os, sys
import cv2
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from vit_pytorch import SimpleViT

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    return faces

class dataBase():
    def __init__(self, cam_id=0):
        self.db_path = './data/'
        self.cam_id = cam_id
        if os.path.exists(self.db_path) == False:
            os.mkdir(self.db_path)

    def take(self, name):
        name_path = self.db_path + name
        cam = cv2.VideoCapture(self.cam_id)
        while True:
            ret, frame = cam.read()
            im = frame.copy()
            
            faces = detect_face(frame)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.putText(frame, 'Q: Exit', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'T: Take picture', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)

            if key == ord('q'):
                sys.exit()

            if key == ord('t'):
                dt = datetime.now().strftime("%m%d%Y%H%M%S")
                face = im[y:y+h, x:x+w]
                H,W = face.shape[1], face.shape[0]
                bg = np.zeros((H+100,W+100,3))
                bg[0:H,0:W] = face/255.0
                cv2.putText(bg, 'Q: Save and exit', (10,H+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(bg, 'R: Return', (10,H+70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('frame', bg)
                key = cv2.waitKey(0)

                if key == ord('q'):
                    if os.path.exists(name_path) == False:
                        os.mkdir(name_path)

                    img_path = name_path + '/' + name + '_' + dt + '.png'
                    cv2.imwrite(img_path, face)
                    break
                
                if key == ord('r'):
                    continue

class detectId():
    def __init__(self, cam_id=0, device='cuda', metric='cosine', threshold_1=0.5, liveness=True, threshold_2=0.6):
        self.cam_id = cam_id
        self.device = device
        self.metric = metric
        self.threshold_1 = threshold_1
        self.liveness = liveness
        self.threshold_2 = threshold_2

        if liveness == True:
            self.liveDetector = SimpleViT(
                image_size = 160,
                patch_size = 32,
                num_classes = 1,
                dim = 1024,
                depth = 6,
                heads = 16,
                mlp_dim = 2048
            )
            self.liveDetector.linear_head[1] = nn.Sequential(nn.Linear(1024,2)) 
            checkpoint = torch.load('./models/model_transformer.pt')
            self.liveDetector.load_state_dict(checkpoint['model'])
            self.liveDetector = self.liveDetector.to(self.device)

        self.latents = []
        self.labels = []
        self.names = os.listdir('./data/')
        self.mtcnn = MTCNN(image_size=160).eval()
        self.model = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()

        if len(self.names) == 0:
            print('No photo found in the database')
            sys.exit()
        else:
            for folder_name in self.names:
                imgs = os.listdir('./data/' + folder_name)
                for img_name in imgs:
                    img_path = './data/' + folder_name + '/' + img_name
                    img = cv2.imread(img_path)
                    img = self.mtcnn(img).unsqueeze(0).to(self.device)
                    latent = self.model(img)[0]
                    self.latents.append(latent)
                    self.labels.append(folder_name)

            self.latents=torch.stack(self.latents)

    def distance(self, x):
        if self.metric == 'cosine':
            norm1 = (torch.sum(x**2))**0.5
            norm2 = (torch.sum(self.latents**2,axis=1))**0.5
            m = 1 - torch.matmul(x, self.latents.T)/(norm1*norm2)
            index = torch.argmin(m[0])

            return index, m[0][index]

        if self.metric == 'euclidian':
            m = [torch.sum(abs(x-self.latents),axis=1)]
            index = torch.argmin(m[0])

            return index, m[0][index]

    def live(self):
        cam = cv2.VideoCapture(self.cam_id)
        while True:
            ret, frame = cam.read()
            
            faces = detect_face(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                face = self.mtcnn(face)
                if face != None:
                    face = face.unsqueeze(0).to(self.device)
                    latent = self.model(face)
                    index, value = self.distance(latent)
                    label = self.labels[index]

                    if value < self.threshold_1:
                        cv2.putText(frame, '%s' % label, (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, 'unknow', (x,y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

                    if self.liveness == False:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, 'None', (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    else:
                        face = 0.5*(face + 1.0)
                        live = self.liveDetector(face)[0][0]
                        if live < self.threshold_2:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, 'Real', (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                            cv2.putText(frame, 'Fake', (x,y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        

            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                sys.exit()

    def faceId(self):
        cam = cv2.VideoCapture(self.cam_id)
        ret, frame = cam.read()
        faces = detect_face(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = self.mtcnn(face)
            if face != None:
                face = face.unsqueeze(0).to(self.device)
                latent = self.model(face)
                index, value = self.distance(latent)
                label = self.labels[index]

                if value > self.threshold_1:
                    label = 'unknown'

        return label
