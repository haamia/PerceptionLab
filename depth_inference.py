from PIL import Image

from models.depth_anything import DepthAnything

image = Image.open("images/uploads/test.jpeg").convert("RGB")

model = DepthAnything()

result = model.estimate(image)

print(result.summary())

print(result.depth_map.shape)