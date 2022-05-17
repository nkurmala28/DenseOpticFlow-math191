import cv2
import numpy as np

# getting the video, making it grayscale, adjusting window size
def draw_hsv(flow,mask):
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    mask[...,0] = ang*(180/np.pi/2)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    return rgb


#setting up the video and preparing the mask to apply ontop of the video
video = cv2.VideoCapture("bruinwalk.mov")
frame_one = video.read()[1]
video_scale = 750/max(frame_one.shape)
frame_one = cv2.resize(frame_one, None, fx=video_scale, fy=video_scale)
prevframe_gray = cv2.cvtColor(frame_one, cv2.COLOR_BGR2GRAY)

out = cv2.VideoWriter('video.mp4', -1, 1, (600, 600))
mask = np.zeros_like(frame_one)
mask[...,1] = 255

#applying the code
while True:
    currframe = video.read()[1]
    currframe_gray = cv2.cvtColor(currframe, cv2.COLOR_BGR2GRAY)
    currframe_gray = cv2.resize(currframe_gray, None, fx=video_scale, fy=video_scale)

    flow = cv2.calcOpticalFlowFarneback(prevframe_gray, currframe_gray, None, 0.5, 8, 10, 5, 5, 1.1, 0)

    rgb = draw_hsv(flow,mask)

    frame = cv2.resize(currframe, None, fx=video_scale, fy=video_scale)

    denseFlow = cv2.addWeighted(frame, 1, rgb, 2, 0)
    cv2.imshow("Gunanr-Farneback", denseFlow)
    out.write(denseFlow)
    prevframe_gray = currframe_gray

    key = cv2.waitKey(5)
    if key == ord('p'):
        cv2.waitKey(-1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
