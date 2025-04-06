import numpy as np
import cv2 as cv

WINDOW_NAME = "Calibration"

WAIT_MSEC = 10
CAPTURE_MSEC = 1000

VIDEO_PATH = "data/chessboard.mp4"

BOARD_PATTERN = (8, 6)
BOARD_CELL_SIZE = 0.025

def capture_timer(count, img, img_select):
    timeout_count = CAPTURE_MSEC // WAIT_MSEC

    if count == timeout_count:
        img_select.append(img)
        # print("Capture timed out.")
        return 0

    return count + 1

def key_event(key, img= None):
    if key == 27:
        return False

    if key == ord(' '):
        if img is not None:
            complete, pts = cv.findChessboardCorners(img, BOARD_PATTERN)
            cv.drawChessboardCorners(img, BOARD_PATTERN, pts, complete)
            cv.imshow(WINDOW_NAME, img)
        cv.waitKey()

    return True

def select_video_and_show():
    video = cv.VideoCapture(VIDEO_PATH)
    assert video.isOpened()

    img_select = []
    capture_time, key = 0, 0
    img = np.array([])
    while key_event(key, img):
        valid, img = video.read()

        if not valid:
            break

        capture_time = capture_timer(capture_time, img, img_select)
        display = img.copy()
        cv.imshow(WINDOW_NAME, display)

        key = cv.waitKey(WAIT_MSEC)

    return img_select

def calibration_camera(images, k= None, d_cf= None):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, BOARD_PATTERN)
        if complete:
            img_points.append(pts)

    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(BOARD_PATTERN[1]) for c in range(BOARD_PATTERN[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * BOARD_CELL_SIZE] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], cameraMatrix=k, distCoeffs=d_cf)

def distortion_correction(k, d_cf):
    video = cv.VideoCapture(VIDEO_PATH)
    assert video.isOpened()

    key = 0
    map1, map2 = None, None
    while key_event(key):
        valid, img = video.read()
        if not valid:
            break

        if map1 is None or map2 is None:
            map1, map2 = cv.initUndistortRectifyMap(k, d_cf, None, None, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        cv.imshow(WINDOW_NAME, img)
        key = cv.waitKey(WAIT_MSEC)

    cv.destroyAllWindows()

if __name__ == '__main__':
    video_file = 'data/chessboard.mp4'

    img_select = select_video_and_show()

    rms, K, dist_coeff, rvecs, tvecs = calibration_camera(img_select)
    distortion_correction(K, dist_coeff)
    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

