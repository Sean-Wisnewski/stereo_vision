import tensorflow_hub as hub

# as suspected, loading from here on the nano takes a century, especially over wifi (and even ethernet)
#model_name ="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
model = hub.load("./downloaded_models/mobilenet_v2")
print(model)