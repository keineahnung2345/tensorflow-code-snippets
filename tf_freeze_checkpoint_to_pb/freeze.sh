# https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#freeze-the-tensorflow-model
# wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
# tar -zxf inception_resnet_v2_2016_08_30.tar.gz
# https://github.com/tensorflow/tensorflow/issues/7172
python3 freeze.py
/usr/local/bin/freeze_graph --input_graph=inception_resnet_v2.pbtxt --input_checkpoint=inception_resnet_v2_2016_08_30.ckpt --output_graph=inception_resnet_v2.pb --output_node_names=InceptionResnetV2/Logits/Predictions
