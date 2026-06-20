import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def align_images(im1, im2):
    """
    Aligns im1 to im2 using ORB feature matching and homography.
    This is critical for scanned PDFs or images that might be slightly shifted.
    """
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Find ORB features and descriptors
    MAX_FEATURES = 5000
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    # Match features using Brute Force Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Keep only the top 15% of matches
    numGoodMatches = int(len(matches) * 0.15)
    matches = matches[:numGoodMatches]

    # Extract coordinates of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography mapping points1 to points2
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography to warp im1 to align with im2
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    return im1Reg

def compare_drawings(old_img_path, new_img_path, output_path="diff_result.jpg", min_contour_area=200):
    """
    Compares two drawing images, highlights differences, and saves the output.
    """
    print(f"Loading '{old_img_path}' and '{new_img_path}'...")
    img1 = cv2.imread(old_img_path)
    img2 = cv2.imread(new_img_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load one or both images. Check file paths.")
        return

    # Resize img1 to match img2 dimensions if they differ fundamentally
    if img1.shape != img2.shape:
        print("Images are different sizes. Resizing old drawing to match new drawing dimensions...")
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    print("Aligning old drawing to new drawing...")
    try:
        img1_aligned = align_images(img1, img2)
    except Exception as e:
        print(f"Alignment failed: {e}. Proceeding without alignment.")
        img1_aligned = img1

    print("Converting to grayscale...")
    gray1 = cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    print("Computing Structural Similarity Index (SSIM)...")
    # Compute SSIM. Returns a score [-1, 1] and a difference image array
    (score, diff) = ssim(gray1, gray2, full=True)
    
    # The diff image contains floating-point values in range [-1, 1]
    # Convert it to 8-bit unsigned integers [0, 255]
    diff = (diff * 255).astype("uint8")

    print(f"Image similarity score: {score:.4f} (1.0 means identical)")

    # Threshold the difference image: differences are represented by darker pixels in the diff map.
    # We invert it so differences become white (255) and similarities become black (0).
    thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]

    # Optional: use morphological operations to close gaps and reduce noise
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    print("Identifying changed regions...")
    # Find contours (outlines) of the differing regions
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes on a copy of the NEW drawing
    output_img = img2.copy()
    changes_found = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        # Filter out tiny differences (noise, compression artifacts)
        if area > min_contour_area:
            x, y, w, h = cv2.boundingRect(c)
            # Draw a red rectangle (BGR format: 0, 0, 255) around the change
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            changes_found += 1

    print(f"Found {changes_found} distinct changed regions.")

    # Save the output visualization
    cv2.imwrite(output_path, output_img)
    print(f"Result saved to: {output_path}")

if __name__ == "__main__":
    # To run this script, provide the paths to your two image files.
    # E.g., if you convert the PDFs to PNGs:
    old_drawing = "old_drawing.png" # Replace with your old drawing image path
    new_drawing = "new_drawing.png" # Replace with your new drawing image path
    
    print("--- Architectural Drawing Diff Tool ---")
    # compare_drawings(old_drawing, new_drawing)
    print("Script is ready. Update the file paths in the script to run the comparison.")
