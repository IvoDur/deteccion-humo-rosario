import cv2

import numpy as np

import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import os

cap = cv2.VideoCapture("humo.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(w, h)

ROJO = (0,0,255)
BLANCO = (255,255,255)

FUENTE = cv2.FONT_HERSHEY_PLAIN
ESCALA_FUENTE = 2
GROSOR_FUENTE = 2 #Pixeles
SEPARACION_TEXTO = w//2
SEPARACION_EJEY_TEXTO = ((h//10) // 5) * 4

desplazamiento = 0
VELOCIDAD_DESPLAZAMIENTO = 1

humo_detectado = False


def alerta(frame,desplazamiento):
    cv2.rectangle(frame, (0,0), (w,h//10), ROJO, -1)
    cv2.rectangle(frame, (0,h), (w,h-h//10), ROJO, -1)

    cv2.putText(frame, "ALERTA", (desplazamiento-w, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento-w+SEPARACION_TEXTO, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento+SEPARACION_TEXTO, h-11), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)

    cv2.putText(frame, "ALERTA", (desplazamiento-w, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento-w+SEPARACION_TEXTO, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)
    cv2.putText(frame, "ALERTA", (desplazamiento+SEPARACION_TEXTO, SEPARACION_EJEY_TEXTO), FUENTE, ESCALA_FUENTE, BLANCO, GROSOR_FUENTE)


#recover our saved model

#generally you want to put the last ckpt from training in here
pipeline_config = 'model\\pipeline.config'
model_dir = 'model\\checkpoint\\ckpt-0'
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(
      model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(
      model=detection_model)
ckpt.restore(os.path.join('model\\checkpoint\\ckpt-0'))


def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""

    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn

detect_fn = get_model_detection_function(detection_model)

#map labels for inference decoding
label_map_path = configs['eval_input_config'].label_map_path
#label_map = label_map_util.load_labelmap(label_map_path)
label_map = label_map_util.load_labelmap("content\\train\\Smoke_label_map.pbtxt")

categories = label_map_util.convert_label_map_to_categories(
    label_map,
    max_num_classes=label_map_util.get_max_label_map_index(label_map),
    use_display_name=True)
category_index = label_map_util.create_category_index(categories)
label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)



while True:

  ref, frame = cap.read()

  frame_correcto = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  input_tensor = tf.convert_to_tensor(
      np.expand_dims(frame_correcto, 0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)

  print(detections['detection_scores'][0].numpy())

  label_id_offset = 1
  image_np_with_detections = frame_correcto.copy()

  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.5,
        agnostic_mode=False,
  )


  frame_mostrar = cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR)

  if humo_detectado:
    alerta(frame_mostrar, desplazamiento)

  cv2.imshow("Camara", frame_mostrar)


  tecla = cv2.waitKey(1)
  if tecla & 0xFF == ord("q"):
    break
  elif tecla & 0xFF == ord("h"):
        humo_detectado = not humo_detectado

  desplazamiento += VELOCIDAD_DESPLAZAMIENTO
  if desplazamiento > w:
      desplazamiento = 0

cap.release()
cv2.destroyAllWindows()

