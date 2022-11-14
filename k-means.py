import cv2
import multiprocessing
import numpy as np


def video2frame(video_name):
    total_frames = []
    cap = cv2.VideoCapture(video_name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps is {}'.format(fps))
    ret, frame = cap.read()
    if cap.isOpened() == False:
        print('Error opening video stream of file')
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            total_frames.append(frame)
        else:
            break
    cap.release()
    return total_frames


def calc_hist(frame):
    h, w, _ = frame.shape
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([temp], [0, 1, 2], None, [12, 5, 5], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= h * w
    return hist


def similarity(a1, a2):
    temp = np.vstack((a1, a2))
    s = temp.min(axis=0)
    si = np.sum(s)
    return si


def ekf(total_frames):
    """extract key frames from total frames.
    First cluster.
    second ekf."""
    centers_d = {}
    result = []
    for i in range(len(total_frames)):
        temp = 0.0
        if len(centers_d) < 1:
            centers_d[i] = [total_frames[i], i]
        else:
            centers = list(centers_d.keys())
            for index, each in enumerate(centers):
                ind = -1
                t_si = similarity(total_frames[i], centers_d[each][0])
                if t_si < 0.5:
                    continue
                elif t_si > temp:
                    temp = t_si
                    ind = index
                else:
                    continue
            if temp > 0.5 and ind != -1:
                centers_d[centers[ind]].append(i)
                length = len(centers_d[centers[ind]]) - 1
                c_old = centers_d[centers[ind]][0] * length
                c_new = (c_old + total_frames[i]) / (length + 1)
                centers_d[centers[ind]][0] = c_new
            else:
                centers_d[i] = [total_frames[i], i]

    cks = list(centers_d.keys())
    for index, each in enumerate(cks):
        if len(centers_d[each]) <= 6:
            result.extend(centers_d[each][1:])
        else:
            temp = []
            accordence = {}
            c = centers_d[each][0]
            for jindex, jeach in enumerate(centers_d[each][1:]):
                accordence[jindex] = jeach

                tempsi = similarity(c, total_frames[jeach])
                temp.append(tempsi)
            oktemp = np.argsort(temp).tolist()
            for i in range(2):
                oktemp[i] = accordence[oktemp[i]]

            result.extend(oktemp[:2])
    return centers_d, sorted(result)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=10)
    video_name = 'video2.mp4'
    total_frames = video2frame(video_name)
    print("there are {} frames in video".format(len(total_frames)))
    h, w, _ = total_frames[0].shape
    hist = pool.map(calc_hist, total_frames)
    cents, results = ekf(hist)
    print(results)
    idx = 0
    cap = cv2.VideoCapture(str(video_name))
    success, frame = cap.read()
    dir = './k2/'
    while (success):
        if idx in results:
            name = "keyframe_" + str(idx) + ".jpg"
            cv2.imwrite(dir + name, frame)
            results.remove(idx)
        idx = idx + 1
        success, frame = cap.read()
    cap.release()
