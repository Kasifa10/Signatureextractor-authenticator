import cv2
import numpy as np
import os

def tightest_signature_crop(image_path, save_path=None, focus_bottom_right=True):
    if not os.path.exists(image_path):
        print(f"❌ File does not exist -> {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Could not read image.")
        return None

    h_img, w_img = img.shape[:2]

    # Focus only on bottom-right (cheques)
    if focus_bottom_right:
        x_start = int(w_img * 0.55)
        y_start = int(h_img * 0.55)
        roi = img[y_start:, x_start:]
    else:
        roi = img.copy()
        x_start, y_start = 0, 0

    # Convert to grayscale and threshold to detect ink
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Light morphological cleaning to preserve gaps
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)

    # Step 1: Find the largest component (likely signature)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        print("❌ No components found.")
        return None

    largest_index = 1 + np.argmax(areas)
    lx, ly, lw, lh, _ = stats[largest_index]

    # Step 2: Keep nearby components (that overlap or are close to largest one)
    mask = np.zeros_like(clean)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # Ignore tiny specks
        if area < 50:
            continue

        # ✂️ Skip printed text-like components (e.g., "Signature", "Please sign above")
        aspect_ratio = w / h if h > 0 else 0
        if (aspect_ratio > 6 and h < 30) or h < 15:
            continue

        # Check if component is close to main signature (in position)
        if (
            x < lx + lw + 20 and x + w > lx - 20 and
            y < ly + lh + 10 and y + h > ly - 10
        ):
            mask[labels == i] = 255

    # Get bounding box of the combined mask
    coords = cv2.findNonZero(mask)
    if coords is None:
        print("❌ No signature region detected.")
        return None

    x, y, w, h = cv2.boundingRect(coords)

    # Map back to full image
    x_full = x + x_start
    y_full = y + y_start

    # Final crop
    cropped = img[y_full:y_full + h, x_full:x_full + w]

    if save_path:
        cv2.imwrite(save_path, cropped)
        print(f"✅ Final tight signature saved to: {save_path}")
    return cropped

# --- Run it ---
if __name__ == "__main__":
    folder = r"C:\\Users\\syedk\\OneDrive\\Documents\\signaturecropping\\cheque"
    input_file = os.path.join(folder, "cheque2.jpg")
    output_file = os.path.join(folder, "final_ultra_signature.png")

    cropped = tightest_signature_crop(input_file, output_file)

    if cropped is not None:
        cv2.imshow("Final Signature", cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
