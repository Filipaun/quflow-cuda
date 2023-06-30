from . import gpu_core

try :
    from . import gpu_tf
except ImportError:
    print("Missing packages for Tensorflow-GPU functionality, use gpu_core instead")
except RuntimeError as e:
    print(e)