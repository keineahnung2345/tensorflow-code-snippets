# tensorflow model freezing

## run
```sh
./freeze.sh
```
This will download `inception_resnet_v2_2016_08_30.tar.gz`, unzip it, and then use `freeze.py` to convert it to `inception_resnet_v2.pbtxt`, finally use the tool `freeze_graph` from tensorflow to generate `inception_resnet_v2.pb`.

Ref: 

[OpenVINO - Freezing Custom Models in Python*](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html#freeze-the-tensorflow-model)

[GitHub issue: freeze graph fail by using the published checkpoint file: Attempting to use uninitialized value](https://github.com/tensorflow/tensorflow/issues/7172)
