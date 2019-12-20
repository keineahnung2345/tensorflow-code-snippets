# tensorflow model freezing

## run
```sh
./freeze.sh
```
This will download `inception_resnet_v2_2016_08_30.tar.gz`, unzip it, and then use `freeze.py` to convert it to `inception_resnet_v2.pbtxt`, finally use the tool `freeze_graph` from tensorflow to generate `inception_resnet_v2.pb`.
