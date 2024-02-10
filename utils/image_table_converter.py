import cv2 as cv
import numpy as np
import argparse


def image_to_table(image_path):
    image = cv.imread(image_path)
    # scale the image to fit the screen
    scale = 900 / image.shape[0]
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

    # let the user select a subimage to crop
    r = cv.selectROI(image)
    imCrop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # get grid lines
    gray = cv.cvtColor(imCrop, cv.COLOR_BGR2GRAY)
    gray = abs(gray - np.ones(gray.shape, np.uint8) * 255)
    edges = cv.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    cv.imshow("Edges", edges)

    lines = cv.HoughLines(edges, 1, np.pi/180, 200)
    print(f"Found {len(lines)} lines")

    vertical_lines = []
    horizontal_lines = []
    for line in lines:
        rho, theta = line[0]
        if np.pi/4 < theta < 3*np.pi/4:
            vertical_lines.append(line)
        else:
            horizontal_lines.append(line)

    # find the intersections
    intersections = []
    for v in vertical_lines:
        for h in horizontal_lines:
            rho_v, theta_v = v[0]
            rho_h, theta_h = h[0]

            A = np.array([
                [np.cos(theta_v), np.sin(theta_v)],
                [np.cos(theta_h), np.sin(theta_h)]
                ])
            b = np.array([rho_v, rho_h])
            x0, y0 = np.linalg.solve(A, b)
            x0, y0 = int(np.round(x0)), int(np.round(y0))
            intersections.append((x0, y0))

    for i in intersections:
        cv.circle(imCrop, i, 5, (0, 0, 255), -1)

    # find cells by sorting the intersections
    intersections.sort(key=lambda x: (x[1], x[0]))

    cv.imshow("Image", imCrop)
    cv.waitKey(0)
    return None


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
            "-i",
            "--image",
            required=True,
            help="Path to the image")
    argParser.add_argument(
            "-o",
            "--output",
            required=True,
            help="Path to the output file")
    args = argParser.parse_args()

    table = image_to_table(args.image)
    print(table)
