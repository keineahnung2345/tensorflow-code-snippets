# tensorflow-code-snippets
Some useful tensorflow code snippets

## List all operations in a graph
```python
import tensorflow as tf
sess = tf.Session()
op = sess.graph.get_operations()
print([m.values() for m in op])
```
