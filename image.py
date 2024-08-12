import cv2

# starting with code from https://medium.com/scrapehero/exploring-image-similarity-approaches-in-python-b8ca0a3ed5a3
def compareHist(image1, image2):
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    hist_img1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img1[255, 255, 255] = 0 #ignore all white pixels
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    hist_img2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist_img2[255, 255, 255] = 0  #ignore all white pixels
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    # Find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
    print(f"Similarity Score: ", round(metric_val, 2))
    # Similarity Score: 0.94

if __name__ == "__main__":
    compareHist("images/hatsune-miku.webp", "images/hatsune-miku.webp")