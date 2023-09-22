# Detector-somnolencia

Sistema de detección de somnolencia en la conducción basado en la monitorización del rostro utilizando técnicas de visión artificial. Se emplea una Raspberry Pi y una cámara ubicada en el interior del vehículo para capturar imágenes faciales en tiempo real. Estas imágenes son procesadas utilizando algoritmos de visión artificial implementados en Python, que extraen características relevantes del rostro. Luego se calculan parámetros clave asociados con la somnolencia como son la apertura de los ojos (EAR) y de la boca (MAR), el porcentaje de tiempo en que los ojos están cerrados (PERCLOS), la posición e inclinación de la cabeza (Head Pose) y la dirección de la mirada (Gaze). Se establecen unos umbrales de alarma que se comparan con los parámetros mencionados para determinar si el conductor se encuentra en un nivel peligroso de somnolencia. En caso de superar dichos umbrales, el sistema activa una alarma para alertar al conductor. Se han considerado cinco alarmas: somnoliento, dormido, distraído, mirada distraída y bostezando.



DESARROLLO DE UN SISTEMA DE VISIÓN ARTIFICIAL PARA DETECTAR Y ALERTAR DE LA SOMNOLENCIA DURANTE LA CONDUCCIÓN MEDIANTE LA MONITORIZACIÓN DE LA CABEZA Y EL ROSTRO

Trabajo Fin de Grado

Grado en Ingeniería Electrónica Industrial y Automática, Universitat Politècnica de València

AUTOR/A: Santos Fernández, Lucas

Tutor/a: Quiles Cucarella, Eduardo

CURSO ACADÉMICO: 2022/2023
