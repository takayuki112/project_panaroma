import cv2
import numpy as np

# Load images
left = cv2.imread('data/input_fragments/room7.jpeg')
right = cv2.imread('data/input_fragments/room6.jpeg')

left = cv2.imread('data/hostel_room_sequence/1room.jpeg')
right = cv2.imread('data/hostel_room_sequence/2room.jpeg')
# target = cv2.imread('data/targets/output.png')

# scale_factor_target = 0.5
# new_size_target = (int(target.shape[1] * scale_factor_target), int(target.shape[0] * scale_factor_target))
# target = cv2.resize(target, new_size_target)

scale_factor_left = 0.5
scale_factor_right = 0.5

new_size_left = (int(left.shape[1] * scale_factor_left), int(left.shape[0] * scale_factor_left))
left = cv2.resize(left, new_size_left)
new_size_right = (int(right.shape[1] * scale_factor_right), int(right.shape[0] * scale_factor_right))
right = cv2.resize(right, new_size_right)


# Convert to grayscale
gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()
sift = cv2.ORB_create()

# Detect keypoints and compute descriptors
keypoints_left, descriptors_left = sift.detectAndCompute(gray_left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(gray_right, None)

# Create BFMatcher and match descriptors
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(descriptors_left, descriptors_right)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Extract matched keypoints' coordinates
points_left = np.array([keypoints_left[m.queryIdx].pt for m in matches])
points_right = np.array([keypoints_right[m.trainIdx].pt for m in matches])

# Estimate homography using RANSAC
H, mask = cv2.findHomography(points_right, points_left, cv2.RANSAC)

# Check if a homography was found
if H is None:
    print("Homography estimation failed.")
else:
    print("Homography estimated successfully.")

# Warp the right image to align with the left image
warped_right = cv2.warpPerspective(right, H, (left.shape[1] + right.shape[1], left.shape[0]))

# Create a blended image
blended = np.zeros((left.shape[0], left.shape[1] + right.shape[1], 3), dtype=np.uint8)

# Place the left image in the blended image
blended[0:left.shape[0], 0:left.shape[1]] = left

# Overlay the warped right image
blended[0:warped_right.shape[0], 0:warped_right.shape[1]] = np.where(warped_right != 0, warped_right, blended[0:warped_right.shape[0], 0:warped_right.shape[1]])

# Visualize matches
matched_image = cv2.drawMatches(left, keypoints_left, right, keypoints_right, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image, stitched result, and target image
cv2.imshow('Matches', right)
cv2.imshow('Stitched Image', blended)
# cv2.imshow('Target', target)

# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Process completed.")
