import cv2

path_image = "zelensky_and_duda.jpeg"
image = cv2.imread(path_image)  # Відкриваємо файл вказаний у змінній path_image
face_cascade = cv2.CascadeClassifier(f"{cv2.data.haarcascades}haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3, minSize=(50, 50))

print(f"Знайдено {len(faces)} облич")

# faces - масив з масивами, що містять координати x і y, довжину та висоти зображення лиця
n = 0
for (x, y, w, h) in faces:
    roi_color = image[y:y + h, x:x + w]
    cv2.imwrite(f"face_{n}_{path_image}", roi_color)  # Зберігаємо зображення
    n += 1
    print("Зображення збережено в папку де знаходиться цей код")
