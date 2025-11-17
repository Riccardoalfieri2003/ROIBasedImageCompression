import cv2
from edges import find_best_edges_by_quality



if __name__ == "__main__":
    image_name = 'images/Hawaii.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    best_edges, best_low, best_high, best_method = find_best_edges_by_quality(image_rgb)

