import cv2
import numpy as np


'''
loss functions
'''
def temporal_loss(x, w, c):
    c = c[np.newaxis, :, :, :]
    D = float(x.size)
    loss = (1. / D) * tf.reduce_sum(c * tf.nn.l2_loss(x - w))
    loss = tf.cast(loss, tf.float32)
    return loss


def get_longterm_weights(i, j):
    c_sum = 0.
    for k in range(args.prev_frame_indices):
        if i - k > i - j:
            c_sum += get_content_weights(i, i - k)
    c = get_content_weights(i, i - j)
    c_max = tf.maximum(c - c_sum, 0.)
    return c_max


def sum_longterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    loss = 0.
    for j in range(args.prev_frame_indices):
        prev_frame = frame - j
        w = get_prev_warped_frame(frame)
        c = get_longterm_weights(frame, prev_frame)
        loss += temporal_loss(x, w, c)
    return loss


def sum_shortterm_temporal_losses(sess, net, frame, input_img):
    x = sess.run(net['input'].assign(input_img))
    prev_frame = frame - 1
    w = get_prev_warped_frame(frame)
    c = get_content_weights(frame, prev_frame)
    loss = temporal_loss(x, w, c)
    return loss

'''
warp image
'''

def warp_image(src, flow):
    w, h, _ = flow.shape
    flow_map = np.zeros(flow.shape, dtype=np.float32)
    for y in range(h):
        flow_map[:, y, 0] += np.float32(y) + flow[:, y, 0]
    for x in range(w):
        flow_map[x, :, 1] += np.float32(x) + flow[x, :, 1]
    # remap pixels to optical flow
    # dst = cv2.remap(
    #     src, flow_map[0], flow_map[1],
    #     interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    print("flow_map:",flow_map)
    dst = cv2.remap(src, flow_map, None, interpolation=cv2.INTER_LINEAR)
    return dst

def preprocess(img):
    # bgr to rgb
    img = img[..., ::-1]
    # shape (h, w, d) to (1, h, w, d)
    img = img[np.newaxis, :, :, :]
    img -= np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))
    return img

cap = cv2.VideoCapture("D:\手机照片2017_1\IMG_1805.MOV")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# while (1):
ret, frame2 = cap.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

w = int(prvs.shape[1])
h = int(prvs.shape[0])
y_coords, x_coords = np.mgrid[0:h, 0:w]
coords = np.float32(np.dstack([x_coords, y_coords]))
pixel_map = coords + flow


# warped_image = cv2.remap(prvs, pixel_map, None, cv2.INTER_LINEAR)
warped_image = warp_image(next, flow).astype(np.float32)


print("coords:",coords)
# print("prev:",prvs)
# print("next:",next)
# print("warped_image:", warped_image)
print(prvs.shape, next.shape, warped_image.shape, flow.shape, coords.shape)
print(np.norm(warped_image - prvs) / np.norm(prvs))

mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang * 180 / np.pi / 2
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('frame1', prvs)
cv2.imshow('frame2', next)
cv2.imshow('frame_warped', warped_image)
k = cv2.waitKey(30) & 0xff
# if k == 27:
#     print("k==27")
# elif k == ord('s'):
#     cv2.imwrite('opticalfb.png', frame2)
#     cv2.imwrite('opticalhsv.png', rgb)
cv2.imwrite('frame1.png', prvs)
cv2.imwrite('frame2.png', next)
cv2.imwrite('frame_warped.png', warped_image)
# prvs = next

cap.release()
cv2.destroyAllWindows()