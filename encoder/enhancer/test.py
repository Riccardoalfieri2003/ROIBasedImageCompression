import cv2
import matplotlib.pyplot as plt

from encoder.enhancer.clahe import get_enhanced_image

if __name__ == "__main__":

    # Load and Convert Image:
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #Enhance Image
    enhanced_rgb=get_enhanced_image(image_rgb)
    


    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_rgb)
    plt.title('Enhanced')
    plt.axis('off')
        
    plt.tight_layout()
    plt.show()