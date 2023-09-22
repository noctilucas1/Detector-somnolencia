import time

class AttentionScorer:

    def __init__(self, capture_fps: int, ear_tresh, gaze_tresh, mar_thresh, perclos_tresh=0.20, ear_time_tresh=4.0, pitch_tresh=165,
                 yaw_tresh=30, gaze_time_tresh=4.0, roll_tresh=20, pose_time_tresh=4.0, verbose=False, mar_time_tresh=3.0):
        """
        La clase Attention Scorer contiene métodos de estimación del EAR, Gaze_Score, PERCLOS y Head Pose a través del tiempo,
        con los thresholds dados (tanto de tiempo como de valores umbrales)

        Parámetros
        ----------
        capture_fps: int
            FPS considerados 

        ear_tresh: float or int
            El valor de EAR threshold (si el EAR es menor que este valor, los ojos son considerados como cerrados)

        mar_thresh: float or int
            El valor de MAR threshold (si el MAR es mayor que este valor, se considera que está bastante abierta la boca)

        gaze_tresh: float or int
            El valor de Gaze threshold (si el Gaze Score es mayor que este valor, la mirada se considera no centrada)

        perclos_tresh: float (ranges from 0 to 1)
            El PERCLOS threshold indica el máximo tiempo permitido en 60 segundos para tener los ojos cerrados sin 
            considerar que hay somnolencia
            (suele utilizarse el 0.2, que sería el 20% de 1 minuto)


        pitch_tresh: int
            Threshold para el ángulo pitch para considerar a una persona distraída 
            (por defecto son 35 grados desde la posición central)

        yaw_tresh: int
            Threshold para el ángulo yaw para considerar a una persona distraída (no teniendo la cabeza orientada hacia el frente)
            (por defecto son 30 grados desde la posición central de la cabeza)

        roll_tresh: int
            Threshold para el ángulo roll para considerar a una persona distraída
            (por defecto es None: no se considera

            
        pose_time_tresh: float or int
            Tiempo máximo permitido para poses de la cabeza distraídas consecutivas 
            (por defecto son 4.0 segundos)

        ear_time_tresh: float or int
            Máximo tiempo permitido para tener los ojos cerrados de manera consecutiva
            (por defecto está a 4.0 segundos)

        mar_time_tresh: float or int
            Máximo tiempo permitido para tener la boca abierta de manera consecutiva

        gaze_time_tresh: float or int
            Máximo tiempo permitido para tener la mirada no centrada

            
        verbose: bool
            Si está a True, escribir información adicional sobre las puntuaciones


        Métodos
        ----------

        - eval_scores: usado para evaluar el estado del conductor 
        - get_PERCLOS: específicamente usado para evaluar la somnolencia del conductor
        """

        self.tiempo_t = 0

        self.fps = capture_fps
        self.delta_time_frame = (1.0 / capture_fps)  # tiempo estimado de un frame
        self.prev_time = 0  # variable auxiliar para la función de estimación del PERCLOS

        self.perclos_time_period = 60 # tiempo por defecto del PERCLOS (60 segundos)
        self.perclos_tresh = perclos_tresh

        # los tresholds de tiempo
        self.ear_tresh = ear_tresh
        self.ear_act_tresh = ear_time_tresh / self.delta_time_frame
        self.ear_counter = 0
        self.eye_closure_counter = 0

        self.gaze_tresh = gaze_tresh
        self.gaze_act_tresh = gaze_time_tresh / self.delta_time_frame
        self.gaze_counter = 0

        self.roll_tresh = roll_tresh
        self.pitch_tresh = pitch_tresh
        self.yaw_tresh = yaw_tresh
        self.pose_act_tresh = pose_time_tresh / self.delta_time_frame
        self.pose_counter = 0

        self.verbose = verbose

        self.mar_tresh = mar_thresh
        self.mar_act_tresh = mar_time_tresh / self.delta_time_frame
        self.mar_counter = 0

    def eval_scores(self, ear_score, gaze_score, head_roll, head_pitch, head_yaw, mar_score):
        """
        :param ear_score: float
            EAR (Eye Aspect Ratio) score obtenido a partir de la apertura de los ojos
        :param gaze_score: float
            Gaze Score obtenido del gaze del ojo
        :param head_roll: float
            Roll ángulo obtenido del head pose 
        :param head_pitch: float
            Pitch ángulo obtenido del head pose 
        :param head_yaw: float
            Yaw ángulo obtenido del head pose 

        :return:
            Devuelve una tupla de valores booleanos que indican el estado del conductor
            tuple: (asleep, looking_away, distracted, yawn)
        """
        # inicialización variables del estado del conductor 
        asleep = False
        looking_away = False
        distracted = False
        yawn = False

        if self.ear_counter >= self.ear_act_tresh:  # comprobar si el EAR acumulativo sobrepasa el threshold
            asleep = True

        if self.gaze_counter >= self.gaze_act_tresh:  # comprobar si el gaze acumulativo sobrepasa el threshold
            looking_away = True

        if self.pose_counter >= self.pose_act_tresh:  # comprobar si el pose acumulativo sobrepasa el threshold
            distracted = True

        if self.mar_counter >= self.mar_act_tresh:  # comprobar si el MAR acumulativo sobrepasa el threshold
            yawn = True

        '''
        Los 3 bloques if que siguen están escritos de manera que cuando tenemos una puntuación que está por encima de su umbral de valor,
        un contador de puntuación respectivo (ear counter, gaze counter, pose counter) se incrementa y puede alcanzar un máximo dado
        con el tiempo.
        Cuando una puntuación no supera un umbral, se disminuye y puede llegar a un mínimo de cero.
        
        Ejemplo:
        
        Si la puntuación del oído del ojo del conductor supera el umbral para un ÚNICO cuadro, el contador de oídos aumenta.
        Si se supera la puntuación de oído del ojo en varios fotogramas, ear_counter aumentará y alcanzará
        un máximo dado, entonces no aumentará, pero la variable "dormido" se establecerá en Verdadero.
        Cuando ear_score no supera el umbral, ear_counter disminuye. Si hay varios marcos
        donde la puntuación no supera el umbral, ear_counter puede alcanzar el mínimo de cero
        
        De esta forma, tenemos una puntuación acumulada para cada una de las características controladas (OÍDO, MIRADA y POSE DE LA CABEZA).
        Si se alcanza una puntuación alta para un contador acumulativo, esta función conservará su valor y necesitará un
        poco de "tiempo de enfriamiento" para volver a cero
        '''
        if (ear_score is not None) and (ear_score <= self.ear_tresh):
            if not asleep:
                self.ear_counter += 1
        elif self.ear_counter > 0:
            self.ear_counter -= 1

        if (gaze_score is not None) and (gaze_score >= self.gaze_tresh):
            if not looking_away:
                self.gaze_counter += 1
        elif self.gaze_counter > 0:
            self.gaze_counter -= 1

        if ((head_pitch is not None and ((head_pitch < -self.pitch_tresh) or ((0 < head_pitch) and (head_pitch < self.pitch_tresh)))) or (
            head_roll is not None and abs(head_roll) > self.roll_tresh) or (
            head_yaw is not None and abs(head_yaw) > self.yaw_tresh)):
            if not distracted:
                self.pose_counter += 1
                None
        elif self.pose_counter > 0:
            self.pose_counter -= 1

        if (mar_score is not None) and (mar_score >= self.mar_tresh):
            if not yawn:
                self.mar_counter += 1
        elif self.mar_counter > 0:
            self.mar_counter -= 1

        if self.verbose:  
            print(
                f"ear counter:{self.ear_counter}/{self.ear_act_tresh}\ngaze counter:{self.gaze_counter}/{self.gaze_act_tresh}\npose counter:{self.pose_counter}/{self.pose_act_tresh}\nmar counter:{self.mar_counter}/{self.mar_act_tresh}")
            print(
                f"eye closed:{asleep}\tlooking away:{looking_away}\tdistracted:{distracted}\yawn:{yawn}")

        return asleep, looking_away, distracted, yawn

    def get_PERCLOS(self, ear_score):
        """

        :param ear_score: float
            EAR (Eye Aspect Ratio) score obtenido a partir de la apertura de los ojos
        :return:
            tuple:(tired, perclos_score)

            tired:
                es un valor booleando que indica si el conductor está cansado o no
            perclos_score:
                es un valor que indica el PERCLOS durante un minuto
                después de un minuto, se resetea el valor
        """

        delta = time.time() - self.prev_time  # calcular delta timer
        tired = False  # inicializar variable de estado del conductor

        # si el EAR es menor o igual que el threshold, incrementar eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_tresh):
            self.eye_closure_counter += 1

        # calcular el closure_time (el tiempo en que los ojos está cerrado) acumulativo
        closure_time = (self.eye_closure_counter * self.delta_time_frame)

        # calcular el PERCLOS en un periodo de tiempo dado
        perclos_score = ((closure_time) / self.perclos_time_period) * 100
               
        if perclos_score >= (self.perclos_tresh*100):  # si el PERCLOS es mayor que el threshold, tired = True
            tired = True

        if self.verbose:
            print(
                f"Closure Time:{closure_time}/{self.perclos_time_period}\nPERCLOS: {round(perclos_score, 3)}")

        if delta >= self.perclos_time_period:  # cada vez que finaliza el periodo de tiempo establecido, se resetea el contador y el timer
            self.eye_closure_counter = 0
            self.prev_time = time.time()

        return tired, perclos_score
    

