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
