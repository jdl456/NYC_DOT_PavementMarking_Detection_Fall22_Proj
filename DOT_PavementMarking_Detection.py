from roboflow import Roboflow
rf = Roboflow(api_key="owxzo5Toq69rSsKwyoKp")
project = rf.workspace().project("pavement-marking-detection")
model = project.version(1).model

# infer on a local image
print(model.predict("aerial_img12.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("aerial_img12.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())