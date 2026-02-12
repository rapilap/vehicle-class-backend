import numpy as np
from PIL import Image
from skimage.feature import hog, local_binary_pattern
import cv2
import io

def preprocess_image(image_bytes, target_size=(128, 128)):
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, target_size)
    img_noise = cv2.GaussianBlur(img_resized, (5, 5), 0)
    img_normalized = img_noise.astype('float32') / 255.0
    img_denorm = (img_normalized * 255).astype('uint8')
    gray_img = cv2.cvtColor(img_denorm, cv2.COLOR_BGR2GRAY)
    
    img_gray_normalized = gray_img.astype('float32') / 255.0
    
    return img_gray_normalized

def extract_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True
    )
    return features

def extract_lbp(image):
    # Convert float image (0-1) ke uint8 (0-255) untuk LBP
    if image.dtype == np.float32 or image.dtype == np.float64:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    features = local_binary_pattern(
        image_uint8,
        P=8,
        R=1,
        method='uniform'
    )
    
    n_bins = 10
    lbp_hist, _ = np.histogram(
        features.ravel(),
        bins=n_bins,
        range=(0, n_bins)
    )
    
    lbp_hist = lbp_hist.astype(float)
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)
    
    return lbp_hist

def extract_fusion_features(image):
    hog_features = extract_hog(image)
    lbp_features = extract_lbp(image)
    
    return {
        'hog': hog_features,
        'lbp': lbp_features
    }