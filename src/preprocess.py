import cv2
import numpy as np
from skimage.morphology import skeletonize
import os


def extract_skeleton(image_path, save_dir="data/skeleton", debug=True):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"❌ Image not found at {image_path}")

    debug_img = img.copy()

    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # ✅ BEST GRAYSCALE VERSION (FOR ML / DEBUG)
    # Illumination normalization + CLAHE
    # --------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    best_gray = clahe.apply(norm)

    cv2.imwrite(os.path.join(save_dir, "gray.png"), best_gray)

    # --------------------------------------------------
    # 1. Blue ink mask
    # --------------------------------------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 2. Light cleanup ONLY
    mask = cv2.medianBlur(mask, 5)

    # --------------------------------------------------
    # 3. BREAK WIRES USING EROSION
    # --------------------------------------------------
    thin_kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask, thin_kernel, iterations=1)

    # --------------------------------------------------
    # 4. COMPONENT DETECTION (ON ERODED IMAGE)
    # --------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        eroded, connectivity=8
    )

    components = []
    wire_mask = mask.copy()

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        blob = (labels == i).astype(np.uint8) * 255
        aspect = max(w, h) / (min(w, h) + 1e-5)
        density = area / (w * h + 1e-5)

        print(
            f"[COMP {i}] area={area}, w={w}, h={h}, "
            f"aspect={aspect:.2f}, density={density:.2f}"
        )

        if area > 200 and density > 0.20 and aspect < 10:
            components.append({
                "bbox": (x, y, w, h),
                "centroid": centroids[i]
            })

            wire_mask[blob > 0] = 0

            if debug:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # --------------------------------------------------
    # 5. SKELETONIZE WIRES ONLY
    # --------------------------------------------------
    wire_skeleton = skeletonize(wire_mask > 0)
    wire_skeleton = (wire_skeleton * 255).astype(np.uint8)

    # --------------------------------------------------
    # 6. Save debug outputs
    # --------------------------------------------------
    cv2.imwrite(os.path.join(save_dir, "components_debug.png"), debug_img)
    cv2.imwrite(os.path.join(save_dir, "wire_mask.png"), wire_mask)
    cv2.imwrite(os.path.join(save_dir, "wire_skeleton.png"), wire_skeleton)

    if debug:
        cv2.imshow("Best Grayscale", best_gray)
        cv2.imshow("Detected Components (RED)", debug_img)
        cv2.imshow("Wire Mask", wire_mask)
        cv2.imshow("Wire Skeleton", wire_skeleton)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return wire_skeleton, components