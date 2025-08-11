import cv2
import numpy as np
import os

def crop_signature(image_path, save_path=None):
    if not os.path.exists(image_path):
        print(f"❌ File does not exist -> {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Threshold for all backgrounds
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )

    # Connect gaps between strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ No signature detected.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    pad = 10
    x = max(x - pad, 0)
    y = max(y - pad, 0)
    w = min(w + 2 * pad, img.shape[1] - x)
    h = min(h + 2 * pad, img.shape[0] - y)

    cropped = img[y:y + h, x:x + w]

    if save_path:
        cv2.imwrite(save_path, cropped)
        print(f"✅ Cropped signature saved to: {save_path}")
    return cropped

# --- Main ---
if __name__ == "__main__":
    # Define your input and output paths here
    folder = r"C:\\Users\\syedk\\OneDrive\\Documents\\signaturecropping\\signatures"
    input_file = os.path.join(folder, "signature1.png")
    output_file = os.path.join(folder, "cropped_signature.png")

    cropped_img = crop_signature(input_file, output_file)

    if cropped_img is not None:
        cv2.imshow("Cropped Signature", cropped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


