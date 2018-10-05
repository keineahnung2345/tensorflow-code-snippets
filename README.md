# tensorflow-code-snippets
Some useful tensorflow code snippets

## List all operations in a graph
```python
import tensorflow as tf
sess = tf.Session()
op = sess.graph.get_operations()
print([m.values() for m in op])
```

## Specify which GPU to use
```python
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"-1" for not using GPU
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
