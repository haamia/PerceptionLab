from PIL import Image

from models.florence2 import Florence2

image = Image.open("images/uploads/test.jpeg").convert("RGB")

model = Florence2()

result = model.caption(image)

print(result.caption)