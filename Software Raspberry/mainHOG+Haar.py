import time
import os

import cv2
import dlib
import numpy as np

from imutils import face_utils
from time import sleep

from picamera import PiCamera
from picamera.array import PiRGBArray

from Utils import get_face_area
from Eye_Mouth_Detector_Module import EyeMouthDetector as EyeMouthDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from Attention_Scorer_Module import AttentionScorer as AttScorer

from Initialization import startup_routine as startrout

CAPTURE_SOURCE = 0 # Webcam

def main():

    t1 = time.time()
    ctime = 0  # tiempo actual (utilizado para calcular los FPS)
    ptime = 0  # tiempo pasado (utilizado para calcular los FPS)
    count_frame = 1
    
    # Adaptar el límite de FPS según la fuente de entrada
    if CAPTURE_SOURCE == 0:
        fps_lim = 3.5 # Para la webcam
    else:
        fps_lim = 25 # Para los vídeos de los dataset

    cv2.setUseOptimized(True)  # Activar la optimización de OpenCV

    # Inicializar el detector de caras de dlib
    DetectorHOG = dlib.get_frontal_face_detector()
    DetectorHaar = cv2.CascadeClassifier(r'/home/lucas/Desktop/Lucas/haarcascade_frontalface_default.xml')

    # Inicializar el predictor de dlib 
    Predictor = dlib.shape_predictor(r'/home/lucas/Desktop/Lucas/shape_predictor_68_face_landmarks.dat')  

    # Instanciar los módulos de detector de ojos y boca, y el estimador de pose
    Eye_mouth_det = EyeMouthDet(show_processing=False)
    Head_pose = HeadPoseEst(show_axis=True)

    # Decisión de activar la rutina de inicialización personalizada
    if CAPTURE_SOURCE != 0:
        MAR_thresh = 0.35
        ear_thresh = 0.2
    
    # Corrección MAR_thresh si la rutina de incialización no sale bien
    if MAR_thresh < 0.35:
        MAR_thresh = 0.35

    Initialization = startrout(DetectorHOG, Predictor, CAPTURE_SOURCE)
    ear_thresh, MAR_thresh = Initialization.personalized_ear_and_mar_thresh()

    #ear_thresh = 0.20
    #MAR_thresh = 0.35

    # Escribir los umbrales de los indicadores MAR y EAR obtenidos
    print("MAR_thresh: " + str(MAR_thresh) + "; EAR_thresh: " + str(ear_thresh))

    # Inicialización del módulo de calcular la puntuación de atención
    Scorer = AttScorer(capture_fps=fps_lim, ear_tresh=ear_thresh, mar_thresh=MAR_thresh, ear_time_tresh=3, gaze_tresh=0.4, perclos_tresh=0.2,
                       gaze_time_tresh=4, pitch_tresh=180, roll_tresh=20, yaw_tresh=30, pose_time_tresh=6, verbose=False, mar_time_tresh=3)

    # Inicializar la captura de imágenes    
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    cap = PiRGBArray(camera, size=(640, 480))
    time.sleep(2)

    # Para crear archivos distintos sucesivos en los que guardar el vídeo y el txt
    i = 0
    Path = "/home/lucas/Desktop/Lucas/ensayo"
    PathAvi = Path + ".avi"
    PathText = Path + ".txt"
    
    while (os.path.exists(PathAvi) == True):
        PathAvi = Path + str(i) + ".avi"
        PathText = Path + str(i) + ".txt"

        i = i + 1       

    # Para grabar el vídeo 
    ancho = 640
    alto = 480
    codec = cv2.VideoWriter_fourcc(*'XVID')
    grabador = cv2.VideoWriter(os.path.abspath(PathAvi), codec, fps_lim, (ancho, alto))

    # Para escribir en un archivo los datos del vídeo
    f = open(os.path.abspath(PathText), 'w')
    #TIEMPO EAR MAR PERCLOS GAZE YAWN SOMNOLENT ASLEEP LOOKING_AWAY DISTRACTED 
    f.write('NFRAME\t FPS\t DETECTOR\t MAR_thresh\t EAR_thresh\t EAR\t MAR\t PERCLOS\t GAZE\t YAWN\t SOMNOLENT\t ASLEEP\t LOOKING_AWAY\t DISTRACTED\t\n')

    # Bucle infinito para capturar los frames con la cámara
    for frames in camera.capture_continuous(cap, format="bgr", use_video_port = True):

        # Leer un frame
        frame = frames.array 

        # Si el frame viene de la webcam, girarlo para que sea como un espejo
        #if CAPTURE_SOURCE == 0:
            #frame = cv2.flip(frame, 2)

        # Calcular los FPS actuales y mostrarlos
        ctime = time.perf_counter()
        fps = 1.0 / float(ctime - ptime)
        ptime = ctime
        cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 150, 50), 2)

        # Transformar el frame en BGR a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar un filtro bilateral para reducir el ruido y resaltar los detalles
        gray = cv2.bilateralFilter(gray, 5, 10, 10)

        # Uso el detector frontal HOG
        faces = DetectorHOG(gray)

        # Para ver usa HOG o Haar
        detectortype = 0

        # Pero si este no encuentra ninguna cara, utilizo el de cvlib (cascadas de Haar)
        
        if len(faces) == 0:
            faces = DetectorHaar.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (30,30))
        
            detectortype = 1

            try:
                # Lo convierto a el formato rectángulo de dlib para luego facilitar el tratamiento
                left, top, right, bottom = faces[0]
                faces = dlib.rectangle(int(left), int(top), int(right), int(bottom)) 
            except:
                None
        

        # Procesa el frame si encuentra al menos una cara
        if faces:

            # Solo cojo la cara más grande, que será la del conductor
            try:
                faces = sorted(faces, key=get_face_area, reverse=True)
                faces = faces[0]
            except:
                None

            # Utilizo el predictor de los 68 puntos clave y lo muestro en el frame
            landmarks = Predictor(frame, faces)
            Eye_mouth_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks)

            #### Para mostrar los contornos ####

            # Crear mapa de puntos
            puntos_coordenadas = face_utils.shape_to_np(landmarks)
            # Se extraen las coordenadas de cada ojo 
            ojo_izq = puntos_coordenadas[42:48]
            ojo_der = puntos_coordenadas[36:42]

            # Se extraen las coordenadas internas de la boca
            mouth = puntos_coordenadas[60:68]

            # Mostrar el contorno de los ojos
            contorno_ojo_izq = cv2.convexHull(ojo_izq)
            contorno_ojo_der = cv2.convexHull(ojo_der)
            cv2.drawContours(frame, [contorno_ojo_izq], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [contorno_ojo_der], -1, (0, 255, 0), 1)
            # Mostrar el contorno de la boca
            contorno_boca = cv2.convexHull(mouth)
            cv2.drawContours(frame,[contorno_boca],-1, (255, 0, 0), 1)

            #### Cálculo de los indicadores y evaluación ####

            # Cálculo del EAR 
            ear = Eye_mouth_det.get_EAR(frame=gray, landmarks=landmarks)

            # Cálculo del MAR
            MAR = Eye_mouth_det.get_MAR(mouth)

            # Calcular el PERCLOS y el estado de somnolencia
            somnolent, perclos_score = Scorer.get_PERCLOS(ear)

            # Calcular el Gaze Score
            gaze = Eye_mouth_det.get_Gaze_Score(frame=gray, landmarks=landmarks)

            # Calcular la posición de la cabeza
            frame_det, roll, pitch, yaw = Head_pose.get_pose(frame=frame, landmarks=landmarks)

            # Evaluar EAR, MAR, GAZE y HEAD POSE
            asleep, looking_away, distracted, yawn = Scorer.eval_scores(
                ear, gaze, roll, pitch, yaw, MAR)  
            
            # Si la estimación de la pose es satisfactoria, mostrar los resultados
            if frame_det is not None:
                frame = frame_det

            # Mostrar EAR 
            if ear is not None:
                cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 110),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
                
            # Mostrar MAR
            if MAR is not None:
                cv2.putText(frame, "MAR:" + str(round(MAR, 3)), (10, 80),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)

            # Mostrar Gaze Score
            if gaze is not None:
                cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 460),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 51), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Gaze Score:0.0", (10, 460),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 51), 1, cv2.LINE_AA)

            # Mostrar PERCLOS 
            cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 1, cv2.LINE_AA)
            
            #### Mostrar evaluaciones ####

            # Si el conductor está somnoliento, mostrar 
            if somnolent:  
                cv2.putText(frame, "SOMNOLIENTO", (10, 260),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Si el conductor está dormido, mostrar y alertar
            if asleep:
                cv2.putText(frame, "DORMIDO", (10, 290),
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Si el conductor tiene la mirada distraída, mostrar
            if looking_away:
                    cv2.putText(frame, "DISTRACCION DE LA MIRADA!", (10, 340),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                    
            # Si el conductor está mirada distraído, mostrar
            if distracted:
                cv2.putText(frame, "DISTRAIDO!", (10, 320),
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                
            # Si el conductor está bostezando, mostrar
            if yawn == True:
                cv2.putText(frame,"BOSTEZANDO", (10, 200), cv2.FONT_HERSHEY_PLAIN, 
                        1.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Escribir en el fichero de texto
            # NFRAME FPS DETECTOR MAR_THRESH EAR_THRESH EAR MAR PERCLOS GAZE YAWN SOMNOLENT ASLEEP LOOKING_AWAY DISTRACTED 
            f.write(str(count_frame) + '\t' + str(fps) + '\t' + str(detectortype) + '\t' + str(MAR_thresh) + '\t' + str(ear_thresh) + '\t' + str(ear) + '\t' + str(MAR) + '\t' + str(perclos_score) + '\t' + str(gaze) + '\t' + str(yawn) + '\t' + str(somnolent) + '\t' + str(asleep) + '\t' + str(looking_away) + '\t' + str(distracted) + '\n')

        # Mostrar el frame por pantalla
        #cv2.imshow("Frame", frame)

        # Guardar frame en el vídeo
        grabador.write(frame)

        count_frame = count_frame + 1 # Contar los frames
        
        cap.truncate(0)
        
        # Sale del bucle si se acaba el tiempo
        if ((time.time() - t1 >= 20*60) or (cv2.waitKey(1) & 0xFF == ord('q'))):
            break

    # Cerrar webcam
    camera.close()
    cv2.destroyAllWindows()

    return

# Llamada al main
if __name__ == "__main__":
    main()
