
import cv2
import time
import statistics

from picamera import PiCamera
from picamera.array import PiRGBArray

from Utils import get_face_area
from Eye_Mouth_Detector_Module import EyeMouthDetector as EyeMouthDet

from imutils import face_utils

class startup_routine():

    def __init__(self, Detector, Predictor, CAPTURE_SOURCE):
        """
        Esta función permite calcular los indicadores EAR y MAR de manera personalizada

        Parámetros
        ----------
        Detector
        Predictor
        CAPTURE_SOURCE

        IMPORTANTE: en esta rutina, se ha de tener la boca abierta para poder calcular bien el MAR
        """

        self.Detector = Detector
        self.Predictor = Predictor
        self.CAPTURE_SOURCE = CAPTURE_SOURCE

        self.ptime = 0
        self.frame_count = 0

        self.ears = []
        self.MARs = []

    def personalized_ear_and_mar_thresh(self):

        # Llamada al detector de ojos y bocas
        Eye_mouth_det = EyeMouthDet(show_processing=False)

        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        cap = PiRGBArray(camera, size=(640, 480))
        time.sleep(2)

        for frames in camera.capture_continuous(cap, format="bgr", use_video_port = True):

            # Leer un frame
            frame = frames.array 

            # Si el frame viene de la webcam, girarlo para que sea como un espejo
            if self.CAPTURE_SOURCE == 0:
                frame = cv2.flip(frame, 2)

            # Calcular los FPS actuales y mostrarlos
            self.ctime = time.perf_counter()
            self.fps = 1.0 / float(self.ctime - self.ptime)
            self.ptime = self.ctime
            cv2.putText(frame, "FPS:" + str(round(self.fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 1)

            # Transformar el frame en BGR a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Aplicar un filtro bilateral para reducir el ruido y resaltar los detalles
            gray = cv2.bilateralFilter(gray, 5, 10, 10)

            # Uso el detector frontal
            faces = self.Detector(gray)

            if faces:  # Procesa el frame si encuentra al menos una cara

                # Solo cojo la cara más grande, que será la del conductor
                faces = sorted(faces, key=get_face_area, reverse=True)
                driver_face = faces[0]

                # Utilizo el preductor de los 68 puntos clave y lo muestro en el frame
                landmarks = self.Predictor(gray, driver_face)
                Eye_mouth_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks)

                # Obtener la apertura de los ojos en posición de reposo
                self.ear = Eye_mouth_det.get_EAR(frame=gray, landmarks=landmarks)
                self.ears.append(self.ear) # Añadir EAR de este frame al array

                # Obtener los puntos de la cara
                puntos_coordenadas = face_utils.shape_to_np(landmarks)

                # Se extraen las coordenadas internas de la boca
                mouth = puntos_coordenadas[60:68]
                
                # Mostrar el contorno de la boca
                contorno_boca = cv2.convexHull(mouth)
                cv2.drawContours(frame,[contorno_boca],-1, (255, 0, 0), 1)

                # Calcular MAR 
                self.MAR = Eye_mouth_det.get_MAR(mouth)
                self.MARs.append(self.MAR) # Añadir MAR de este frame al array

                # Mostrar frame
                #cv2.imshow("Frame", frame)

            # Sumar uno al conteo de frames
            self.frame_count += 1
            
            cap.truncate(0)

            # Salir del bucle si se pulsa la tecla "q" o si pasan 50 frames (unos 3-4 segundos, depende de los FPS)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or self.frame_count > 20:
                
                # Cuando pasen x frames, se corta el vídeo y se calcula la media 
                mean_ear = statistics.mean(self.ears)
                mean_MAR = statistics.mean(self.MARs)

                # Calculamos el ear_thresh como 3/4 partes de la media, para evitar tener en cuenta los parpadeos
                ear_thresh = mean_ear*(3/4)

                # Calculamos el MAR_thresh como la mitad de la media estadística, ya que se tomará con la boca abierta, y así
                # consideraremos que a partir de la mitad de apertura (y durante determinado tiempo), es un bostezo
                MAR_thresh = mean_MAR/2

                break

        camera.close()
        cv2.destroyAllWindows()

        return ear_thresh, MAR_thresh

