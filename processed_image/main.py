import cv2
import numpy as np

image = cv2.imread("rabbit.jpeg")
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])


def sum_of_area(matrix1, matrix2):
    sum = matrix1 * matrix2
    sum = np.sum(sum, axis=1)
    sum = np.sum(sum, axis=0)
    return sum


def convolution(matrix, kernel):
    global row, column
    # padding = 1
    matrix = np.hstack((np.zeros((row, 1)), matrix, np.zeros((row, 1))))
    matrix = np.vstack((np.zeros((1, column + 2)), matrix, np.zeros((1, column + 2))))
    new_image = np.array(np.zeros((row, column)), dtype=np.uint8)

    for i in range(row):
        for j in range(column):
            m = matrix[i:i+3, j:j+3]
            new_image[i][j] = sum_of_area(m, kernel)

    return new_image


image_r = image[:, :, 0]
image_g = image[:, :, 1]
image_b = image[:, :, 2]

image_bw = 0.299*image_r + 0.587*image_g + 0.114*image_b
image_bw = np.array(image_bw, np.uint8)
row, column = image_bw.shape


B = convolution(image_b, kernel)
G = convolution(image_g, kernel)
R = convolution(image_r, kernel)

new_image = cv2.merge([R, G, B])

cv2.imshow('image demo', new_image)

cv2.waitKey()








