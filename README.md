# Phishing Classifier

En una sociedad donde prima el uso de la tecnología en cualquier tarea cotidiana, los ataques informáticos están a la orden del día. Uno de los ataques más comunes y con mayor tasa de éxito actualmente es el phishing. Este es un tipo de fraude en el que un atacante suplanta la identidad a una entidad legítima, generalmente, con el fin de obtener información personal y de tarjetas bancarias de cualquier persona por medio de correos electrónicos, mensajes de texto o redes sociales. Independientemente del medio por el que se lleve a cabo, el atacante acostumbra a redirigir a las potenciales víctimas a una página web que simula ser legítima, engañándolas para recoger información de carácter personal o conseguir que paguen por un servicio que nunca recibirán. Además, dicho fraude también puede ser utilizado como vehículo para otros tipos de ataque, como podría ser el _ransomware_.

El alto éxito de los ataques de phishing suele ser debido a la baja complejidad de ejecución de los mismos, a la existencia de herramientas que facilitan aún más la tarea de la suplantación y clonación de las páginas web, o a la falta de concienciación en materia de ciberseguridad por parte de ciertos grupos de la población, entre los que destacan las personas mayores y los niños pequeños. Debido a lo anterior, se considera necesario el desarrollo de algoritmos y sistemas de detección automática de páginas web fraudulentas que sirvan como apoyo para toda la población.

En este Trabajo Fin de Grado se van a estudiar y aplicar distintas técnicas de inteligencia artificial para la clasificación de páginas web con el objetivo de mejorar el rendimiento de los algoritmos existentes y de cara a proporcionar una clasificación más precisa de las webs. Para ello, se van a implementar distintos algoritmos de aprendizaje automático, como regresión logística o _random forest_, y de aprendizaje profundo, como redes neuronales convolucionales.

El repositorio contiene los seis experimentos que se han llevado a cabo en este Trabajo Fin de Grado:
1. Ejecución de modelos clásicos de machine learning: 
2. Escalado de datos del dataset y repetición del experimento 1.
3. Ensemble de votación con regresión logística, _random forest_ y SVM.
4. Red neuronal con una capa oculta.
5. Red con capas LSTM.
6. Red neuronal convolucional.

También se proporciona el código para generar las imágenes del experimento 6 y el código para analizar las características más significativas para la clasificación de páginas web mediante _eXplainable Artificial Intelligente_ (XAI). De igual forma, se aportan las imágenes ya generadas.