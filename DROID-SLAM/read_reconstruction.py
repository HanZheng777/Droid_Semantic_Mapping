import numpy as np
import cv2

depth = np.load("reconstructions/disps.npy")
image = np.load("reconstructions/images.npy")

max_depth = 3

depth = 1/np.clip(depth, 1/max_depth, None)

depth_norm = cv2.normalize(depth, None, 0,1,cv2.NORM_MINMAX)
depth_scaled = cv2.convertScaleAbs(depth_norm, alpha=255.0)




print(depth_scaled[0])
# print(image.shape)


# Load the numpy array from the npy file
arr = depth_scaled
# arr = image
# Define the output video file name and parameters
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_result/output.mp4', fourcc, 25, (560, 344))

# iterate over each frame in the array and write it to the video file
for i in range(arr.shape[0]):
    # convert the frame to a 8-bit unsigned integer format
    frame = np.uint8(arr[i])
    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    # transpose the frame to (height, width, channels) format expected by OpenCV
    # frame = np.transpose(frame, (1, 2, 0))
    # write the frame to the video file
    out.write(frame)

# release the video writer and close the video file
out.release()

