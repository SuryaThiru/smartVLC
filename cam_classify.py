import pickle
import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure


def apply_hog(img, visualise=True):
    fd, hog_im = hog(img, orientations=9, pixels_per_cell=(5, 5),
                    cells_per_block=(1, 1), visualise=visualise,
                    block_norm='L2-Hys')

    hog_image_rescaled = exposure.rescale_intensity(hog_im, in_range=(0, 10))
    return hog_image_rescaled


def add_disk_mask(im):
    r, c = im.shape
    rw, cl = np.ogrid[:r, :c]
    cr, cc = r / 2, c / 2

    mask = (((rw - cr) ** 2) + ((cl - cc) ** 2) > (cr ** 2))
    im[mask] = 0

    return im


def flatten(img):
    return img.reshape(1, img.shape[0] * img.shape[1])


def preprocessing_pipeline(image):
    print('processing...')
    final = apply_hog(image)
    final = add_disk_mask(final)
    final = flatten(final)

    return final


def capture_image():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    key = cv2.waitKey(1)
    cap.release()

    return frame


def classify_image(clf, im):
    y = clf.predict(im)
    if y == 0:
        return 'none'
    elif y == 1:
        return 'play'
    elif y == 2:
        return 'pause'
    elif y == 3:
        return 'volumn down'
    elif y == 4:
        return 'volumn up'


clf = pickle.load(open('model.json', 'rb'))
im = capture_image()
im = preprocessing_pipeline(im)
label = classify_image(clf, im)
print(label)
