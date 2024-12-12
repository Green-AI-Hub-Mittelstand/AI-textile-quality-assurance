import tensorflow as tf
import tf2onnx
import onnx

# Load and Save files to adapt.
h5_model_path = "model_filepath.../resmodel50_tf_classifier_p_560.h5"
save_path = "./model.onnx"

tf_model = tf.keras.models.load_model(h5_model_path)
tf_model.summary()
onnx_model, _ = tf2onnx.convert.from_keras(tf_model,[tf.TensorSpec(
            tf_model.inputs[0].shape,
            dtype=tf_model.inputs[0].dtype,
            name=tf_model.inputs[0].name,)])

onnx.save_model(onnx_model,save_path)
