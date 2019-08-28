import requests
import numpy as np
import cv2

url = "https://chart.googleapis.com/chart?chs=500x500&cht=qr&chl=123456789012&chld=H"
response = requests.get(url)
if response.status_code == 200:
    with open("../tmp/sample.jpg", 'wb') as f:
        r=response.content
        f.write(response.content)
        image = np.asarray(bytearray(r), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite("../tmp/op.jpg", image)