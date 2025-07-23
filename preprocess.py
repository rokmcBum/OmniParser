import cv2
import numpy as np
from PIL import Image


def preprocess_for_ocr(image_input: Image.Image, apply_sharpen=True, apply_denoise=True,
                       apply_binarize=True) -> Image.Image:
    """
    PIL 이미지를 받아서 샤프닝, 노이즈 제거, 이진화까지 수행한 후 다시 PIL 이미지로 반환
    """
    # 1. PIL → OpenCV (BGR)
    img_rgb = image_input.convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)

    # 2. Sharpen (선택적)
    if apply_sharpen:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        img_bgr = cv2.filter2D(img_bgr, -1, kernel)

    # 3. Denoise (선택적)
    if apply_denoise:
        img_bgr = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # 4. Convert to Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 5. Binarization (선택적)
    if apply_binarize:
        # adaptive thresholding (조명 변화에 강함)
        binarized = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # 또는 ADAPTIVE_THRESH_MEAN_C
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
    else:
        binarized = gray

    # 6. OpenCV → PIL
    result_pil = Image.fromarray(binarized)

    return result_pil
