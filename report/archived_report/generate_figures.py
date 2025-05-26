from os import system
from scipy.datasets import face as get_face
from imageio.v3 import imwrite
from PIL import Image
import numpy as np

def main():
    face = get_face(gray=True)
    face = Image.fromarray(face).resize((256, 256))
    imwrite("face.bmp", np.array(face))
    system("cargo run encode face.bmp face.rsarc 4096 5")

    rng = np.random.default_rng(42)
    def randomize(arr):
        for i in range(0, len(arr)):
            arr[i] = rng.integers(0, 256)

    face = np.array(face)
    face_2 = face.copy()
    face_3 = face.copy()
    randomize(face_2.ravel()[7300:20300])

    for i in range(0, face_3.size, 2000):
        randomize(face_3.ravel()[i:i+100])

    print(f"Differences between face and face_2: {np.sum(face != face_2)}")
    print(f"Differences between face and face_3: {np.sum(face != face_3)}")

    imwrite("face_2.bmp", face_2)
    imwrite("face_2_repaired.bmp", face_2)
    imwrite("face_3.bmp", face_3)
    imwrite("face_3_repaired.bmp", face_3)

    system("cargo run repair face_2_repaired.bmp face.rsarc")
    system("cargo run repair face_3_repaired.bmp face.rsarc")

    face_2_repaired = Image.open("face_2_repaired.bmp")
    face_3_repaired = Image.open("face_3_repaired.bmp")
    assert np.all(np.array(face_2_repaired) == face)
    assert not np.all(np.array(face_3_repaired) == face)

    def to_png(path):
        imwrite(f"{path}.png", np.array(Image.open(f"{path}.bmp")))

    to_png("face_2")
    to_png("face_3")
    to_png("face_2_repaired")

if __name__ == "__main__":
    main()
