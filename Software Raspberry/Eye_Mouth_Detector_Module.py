import cv2
import numpy as np
from numpy import linalg as LA
from Utils import resize
from scipy.spatial import distance as calculo_distancia

class EyeMouthDetector:

    def __init__(self, show_processing: bool = False):
        """
        La clase Eye detector contiene varios metodos para estimar la apertura de los ojos, de la boca y del gaze

        Parámetros 
        ----------
        show_processing: bool
            Si se pone a True, muestra los frames durante el procesamiento en algunos pasos

        Métodos 
        ----------
        - show_eye_keypoints: muestra los keypoints de los ojos en el frame
        - get_EAR: calcula el parámetro EAR para los dos ojos
        - get_MAR: calcula el parámetro MAR para la boca
        - get_Gaze_Score: calcula el Gaze (la distancia euclídea normalizada entre el centro del ojo y la pupila)
        """

        self.keypoints = None
        self.frame = None
        self.show_processing = show_processing
        self.eye_width = None

    def show_eye_keypoints(self, color_frame, landmarks):
        """
        Muestra los keypoints de los ojos encontrados en la cara, dibujando círculos rojos en esas posiciones del frame

        Parámetros
        ----------
        color_frame: numpy array
            Frame a color 
        landmarks: list
            Lista de los 68 dlib keypoints de la cara
        """

        self.keypoints = landmarks

        for n in range(36, 48):
            x = self.keypoints.part(n).x
            y = self.keypoints.part(n).y
            cv2.circle(color_frame, (x, y), 1, (0, 0, 255), -1)
        return

    def get_EAR(self, frame, landmarks):
        """
        Calcula el eye aperture rate de la cara

        Parámetros
        ----------
        frame: numpy array
            Frame a color
        landmarks: list
            Lista de los 68 dlib keypoints de la cara

        Returns
        -------- 
        ear_score: float
            EAR average entre los dos ojos
            El EAR o Eye Aspect Ratio se calcula como la apertura del ojo dividida entre la longitud del ojo
            Cada ojo tiene una puntuación y las dos puntuaciones se promedian
        """

        self.keypoints = landmarks
        self.frame = frame
        pts = self.keypoints

        i = 0  # contador auxiliar
        # array para almacenar las posiciones de los keypoints del ojo izquierdo
        eye_pts_l = np.zeros(shape=(6, 2))
        # array para almacenar las posiciones de los keypoints del ojo derecho
        eye_pts_r = np.zeros(shape=(6, 2))

        for n in range(36, 42):  # los dlib keypoints del 36 al 42 se refieren al ojo izquierdo
            point_l = pts.part(n)  # guardar los i-keypoint del ojo izquierdo
            point_r = pts.part(n + 6)  # guardar los i-keypoint del ojo derecho
            # array de las coordenadas x,y para el punto de referencia del ojo izquierdo
            eye_pts_l[i] = [point_l.x, point_l.y]
            # array de las coordenadas x,y para el punto de referencia del ojo derecho
            eye_pts_r[i] = [point_r.x, point_r.y]
            i += 1  # incrementar contador 

        def EAR_eye(eye_pts):
            """
            Calcular el EAR score para un solo ojo dados sus keypoints
            :param eye_pts: numpy array of shape (6,2) que contienen los keypoints de un ojo considerando dlib
            :return: ear_eye
                EAR del ojo
            """
            ear_eye = (LA.norm(eye_pts[1] - eye_pts[5]) + LA.norm(
                eye_pts[2] - eye_pts[4])) / (2 * LA.norm(eye_pts[0] - eye_pts[3]))
            '''
            EAR se calcula como la media de las dos medidas de apertura del ojo dividido entre la longitud del ojo
            '''
            return ear_eye

        ear_left = EAR_eye(eye_pts_l)  # calcular EAR para el ojo izquierdo
        ear_right = EAR_eye(eye_pts_r)  # calcular EAR para el ojo derecho

        # calcular la media del EAR
        ear_avg = (ear_left + ear_right) / 2

        return ear_avg

    def get_MAR(self, mouth):
        """
        Calcular el mouth aspect ratio (MAR)
        Se establece la relación entre la longitud y el ancho de la región de la boca

        Parámetros
        ----------
        mouth: numpy array
            Keypoints de la boca 

        Returns
        -------
        MAR: float
            Puntuación del MAR
        """

        # calcular la distancia entre los puntos que definen la posición de la boca interna según el dlib68
        A = calculo_distancia.euclidean(mouth[1], mouth[7])
        B = calculo_distancia.euclidean(mouth[2], mouth[6])
        C = calculo_distancia.euclidean(mouth[3], mouth[5])  
        D = calculo_distancia.euclidean(mouth[0], mouth[4])  
        # Cálculo MAR
        MAR = (A+B+C)/2.0/D
        return MAR

    def get_Gaze_Score(self, frame, landmarks):
        """
        Calcular el promedio Gaze para los ojos
        El Gaze Score es la media de la norma l2 (distancia euclidiana) entre el punto central de la ROI del ojo 
        (la bounding box del ojo) y el centro de la pupila

        Parámetros
        ----------
        frame: numpy array
            Frame en el que se encuentran los ojos
        landmarks: list
            Lista de los 68 dlib keypoints de la cara

        Returns
        -------- 
        avg_gaze_score: float
            If successful, returns the float gaze score
            Si no es cero, devuelve la puntuación calculada
            Si es cero, devuelve None
        """
        self.keypoints = landmarks
        self.frame = frame

        def get_ROI(left_corner_keypoint_num: int):
            """
            Obtiene la ROI bounding box del ojo a partir del keypoint del ojo

            :param left_corner_keypoint_num:  el keypoint del ojo más a la izquierda
            :return: eye_roi
                Sub-frame de la región del ojo del frame
            """

            kp_num = left_corner_keypoint_num

            eye_array = np.array(
                [(self.keypoints.part(kp_num).x, self.keypoints.part(kp_num).y),
                 (self.keypoints.part(kp_num+1).x,
                  self.keypoints.part(kp_num+1).y),
                 (self.keypoints.part(kp_num+2).x,
                  self.keypoints.part(kp_num+2).y),
                 (self.keypoints.part(kp_num+3).x,
                  self.keypoints.part(kp_num+3).y),
                 (self.keypoints.part(kp_num+4).x,
                  self.keypoints.part(kp_num+4).y),
                 (self.keypoints.part(kp_num+5).x, self.keypoints.part(kp_num+5).y)], np.int32)

            min_x = np.min(eye_array[:, 0])
            max_x = np.max(eye_array[:, 0])
            min_y = np.min(eye_array[:, 1])
            max_y = np.max(eye_array[:, 1])

            eye_roi = self.frame[min_y-2:max_y+2, min_x-2:max_x+2]

            return eye_roi

        def get_gaze(eye_roi):
            """
            Calcula la norma L2 entre el punto central de la ROI del ojo y el centro de la pupila
            :param eye_roi: float
            :return: (gaze_score, eye_roi): tuple
                tuple
            """

            eye_center = np.array(
                [(eye_roi.shape[1] // 2), (eye_roi.shape[0] // 2)])  # posición central de la ROI del ojo
            gaze_score = None
            circles = None

            # filtro bilateral para reducir el sonido y resaltar los detalles
            if eye_roi.any():
                eye_roi = cv2.bilateralFilter(eye_roi, 4, 40, 40)

                circles = cv2.HoughCircles(eye_roi, cv2.HOUGH_GRADIENT, 1, 10,
                                        param1=90, param2=6, minRadius=1, maxRadius=9)
            # Transformada de Hough para encontrar el iris y su centro (la pupila) en la eye_roi de la imagen en escala de grises 

            if circles is not None and len(circles) > 0:
                circles = np.uint16(np.around(circles))
                circle = circles[0][0, :]

                cv2.circle(
                    eye_roi, (circle[0], circle[1]), circle[2], (255, 255, 255), 1)
                cv2.circle(
                    eye_roi, (circle[0], circle[1]), 1, (255, 255, 255), -1)

                # la posición de la pupula es el primer círculo que se encuentra con la transformada de Hough
                pupil_position = np.array([int(circle[0]), int(circle[1])])

                cv2.line(eye_roi, (eye_center[0], eye_center[1]), (
                    pupil_position[0], pupil_position[1]), (255, 255, 255), 1)

                gaze_score = LA.norm(
                    pupil_position - eye_center) / eye_center[0]
                # calcula la L2 distancia entre el centro del ojo y el centro de la pupila

            cv2.circle(eye_roi, (eye_center[0],
                                 eye_center[1]), 1, (0, 0, 0), -1)

            if gaze_score is not None:
                return gaze_score, eye_roi
            else:
                return None, None

        left_eye_ROI = get_ROI(36)  # calcula la ROI para el ojo izquierdo
        right_eye_ROI = get_ROI(42)  # calcula la ROI para el ojo derecho

        # calcula el gaze para los ojos
        gaze_eye_left, left_eye = get_gaze(left_eye_ROI)
        gaze_eye_right, right_eye = get_gaze(right_eye_ROI)

        # si show_processing es True, muestra la ROI de los ojos, el centro del ojo, el centro de la pupila y las líneas de distancia
        if self.show_processing and (left_eye is not None) and (right_eye is not None):
            left_eye = resize(left_eye, 1000)
            right_eye = resize(right_eye, 1000)
            cv2.imshow("left eye", left_eye)
            cv2.imshow("right eye", right_eye)

        if gaze_eye_left and gaze_eye_right:

            # calcula el promedio del gaze score para los 2 ojos
            avg_gaze_score = (gaze_eye_left + gaze_eye_left) / 2
            return avg_gaze_score

        else:
            return None
