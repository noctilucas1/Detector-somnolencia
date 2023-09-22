import cv2
import numpy as np

from Utils import rotationMatrixToEulerAngles, draw_pose_info


class HeadPoseEstimator:

    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):
        """
        Head Pose estimator contiene el método get_pose para calcular los 3 ángulos de Euler (roll, pitch, yaw)
        de la cabeza. Usa el frame, los landmarks de la cabeza detectados con dlib y, opcionalmente, los 
        parámetros de la cámara

        Parámetros
        ----------
        camera_matrix: numpy array
            La matriz de la cámara usada
        dist_coeffs: numpy array
            Los coeficientes de distorsión de la cámara
        show_axis: bool
            Para enseñar los ejes proyectados sobre la nariz
        """

        self.verbose = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def get_pose(self, frame, landmarks):
        """
        Estima la posición de la cabeza usando el estimador de pose 

        Parámetros
        ----------
        frame: numpy array
            Frame capturado por la cámara
        landmarks: dlib.rectangle
            Los 68 landkmarks de la cabeza detectados con dlib

        Devuelve
        --------
        - if successful: image_frame, roll, pitch, yaw (tuple)
        - if unsuccessful: None,None,None,None (tuple)

        """
        self.keypoints = landmarks  # dlib 68 landmarks
        self.frame = frame  

        self.axis = np.float32([[200, 0, 0],
                                [0, 200, 0],
                                [0, 0, 200]])
        # array que especifica el largo de los 3 ejes que se proyectan sobre la nariz

        if self.camera_matrix is None:
            # Si no hay una matriz de la cámara, estimar los parámetros utilizando un frame
            self.size = frame.shape
            self.focal_length = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )

        if self.dist_coeffs is None:  # Asumir que no hay distorsión de lente si no se proporcionan los coeficientes de distorsión
            self.dist_coeffs = np.zeros((4, 1))

        # Modelo 3D de la cabeza génerica humana
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nariz
            (0.0, -330.0, -65.0),  # Mentón
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
            ])

        # Posición 2D de los keypoints faciales de dlib usados para la estimación de la pose
        self.image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nariz
            (landmarks.part(8).x, landmarks.part(8).y),  # Mentón
            (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corner
            (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

        # Calcular la pose de la cabeza
        (success, rvec, tvec) = cv2.solvePnP(self.model_points, self.image_points,
                                             self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        '''
        OpenCV Solve PnP calcula los vectores de rotación y traslación con respecto al sistema de coordenadas de la cámara
        con los puntos de la imagen que tienen de referencia el modelo 3D de la cabeza
        '''

        if success:  # Si funciona, calcular la posición de la cabeza

            # Refinar rvec y tvec
            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec)

            # Punto de la nariz en la imagen
            nose = (int(self.image_points[0][0]), int(self.image_points[0][1]))

            # Calcular la proyección de los 3 ejes desde la nariz
            (nose_end_point2D, _) = cv2.projectPoints(self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)

            # Calcular la matriz de rotación a partir del vector de rotación con la fórmula Rodrigues
            Rmat = cv2.Rodrigues(rvec)[0]

            pitch, yaw, roll = rotationMatrixToEulerAngles(Rmat) * 180/np.pi

            """
            La función rotationMatrixToEulerAngles se utiliza para calcular los ángulos de euler a partir de la 
            matriz de rotación. Los ángulos se convierten a radianes.
            """

            # Para mostrar los datos de la función 
            if self.verbose:
                self.frame = draw_pose_info(
                    self.frame, nose, nose_end_point2D, roll, pitch, yaw)
                # draws 3d axis from the nose and to the computed projection points
                for point in self.image_points:
                    cv2.circle(self.frame, tuple(
                        point.ravel().astype(int)), 2, (0, 255, 255), -1)
                # Dibuja los 6 keypoints usados para la pose estimation

            return self.frame, roll, pitch, yaw

        else:
            return None, None, None, None
