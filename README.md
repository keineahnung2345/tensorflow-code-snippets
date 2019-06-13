# tensorflow-code-snippets
Some useful tensorflow code snippets

## List all operations in a graph
```python
import tensorflow as tf
sess = tf.Session()
op = sess.graph.get_operations()
print([m.name for m in op]) #print the name of operations
print([m.values() for m in op]) #print the tensor produced by these operations
```

## List all tensors in a graph
```python
print([n.name for n in tf.get_default_graph().as_graph_def().node])
```

## Get tensor by name
```python
tf.get_default_graph().get_tensor_by_name("<your-tensor-name>:0")
```

## Find out the operation that generates this tensor
```python
<your-tensor>.op
```

## Find out the operands that builds up this tensor
```python
<your-tensor>.op.inputs
```

## Check how many operands building up this tensor
```python
<your-tensor>.op.inputs.__len__()
```

## Zero out some entries of tensor x according to a boolean mask
```python
x = tf.where(mask, x, tf.zeros_like(x))
```

## Get visible GPUs
```python
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()
```

## keras: Get visible GPUs
```python
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
```

## Specify which GPU to use
```python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"-1" for not using GPU
```

## Clear session, destroy the current graph and create a new one
```python
import tensorflow as tf
tf.keras.backend.clear_session()
```

## keras: specify the proportion of GPU to use
```python
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

def get_session(gpu_fraction=0.333):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=gpu_fraction,
        allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

ktf.set_session(get_session())
```

## keras: show all layers of a model
```python
model.layers
```

## keras: get layer of a model
```python
model.get_layer(<"layer-name">)
model.get_layer(index=<layer-index>)
```

## command line: open tensorboard on your-host-ip:6006
To use tensorboard with PyTorch, first:
```sh
pip install tensorflow tensorboardX
```
And then:
```sh
tensorboard --logdir=<your-log-dir> --port=<your-port>
```
