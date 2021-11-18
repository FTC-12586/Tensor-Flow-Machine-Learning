import tensorflow as tf


class YoloReshape(tf.keras.layers.Layer):
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def build(self, input_shape):
        pass

    def call(self, inputs, **kwargs):
        pass

    def get_config(self):
        return self.__dict__

    """  * `__init__()`: Defines custom layer attributes, and creates layer state
    variables that do not depend on input shapes, using `add_weight()`.
  * `build(self, input_shape)`: This method can be used to create weights that
    depend on the shape(s) of the input(s), using `add_weight()`. `__call__()`
    will automatically build the layer (if it has not been built yet) by
    calling `build()`.
  * `call(self, *args, **kwargs)`: Called in `__call__` after making sure
    `build()` has been called. `call()` performs the logic of applying the
    layer to the input tensors (which should be passed in as argument).
    Two reserved keyword arguments you can optionally use in `call()` are:
      - `training` (boolean, whether the call is in
        inference mode or training mode)
      - `mask` (boolean tensor encoding masked timesteps in the input, used
        in RNN layers)
  * `get_config(self)`: Returns a dictionary containing the configuration used
    to initialize this layer. If the keys differ from the arguments
    in `__init__`, then override `from_config(self)` as well.
    This method is used when saving
    the layer or a model that contains this layer."""
