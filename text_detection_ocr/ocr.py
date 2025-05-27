import cv2
import easyocr
import matplotlib.pyplot as plt

image_path = 'text_detection_ocr/test_image.png'
img = cv2.imread(image_path)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

reader = easyocr.Reader(['en'], gpu=False)  # Initialize the reader with English language and CPU mode
text = reader.readtext(img)

for t in text:
    bbox, text_content, confidence = t
    cv2.rectangle(img, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[2][0]), int(bbox[2][1])), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()