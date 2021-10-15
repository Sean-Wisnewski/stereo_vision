import tensorflow_hub as hub

model_name ="https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
model = hub.load(model_name)
print(model)