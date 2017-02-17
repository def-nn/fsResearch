import math
import numpy as np
import cv2
import time
import pymeanshift


DIM_NUM = 6
IND_SIZE = 0
IND_SPATIAL = 1
IND_RANGE = 3
IND_DIAMETER = 4


def RGB_to_Luv(b, g, r):
    b /= 255
    g /= 255
    r /= 255

    b = ((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92
    g = ((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92
    r = ((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92

    b *= 100
    g *= 100
    r *= 100

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505

    u = (4 * x) / (x + (15 * y) + (3 * z))
    v = (9 * y) / (x + (15 * y) + (3 * z))

    y /= 100
    y = y ** (1 / 3) if y > 0.008856 else (7.787 * y) + (16 / 116)

    _x = 95.047
    _y = 100.000
    _y = 108.883

    _u = (4 * _x) / (_x + (15 * _y) + (3 * _y))
    _v = (9 * _y) / (_x + (15 * _y) + (3 * _y))

    L = (116 * y) - 16
    u = 13 * L * (u - _u)
    v = 13 * L * (v - _v)

    return L, u, v


def compute_kernel(dist, dimension, h):
    if dist >= h: return 0
    c = {
        1: 2 * math.pi * h,
        2: math.pi * h ** 2,
        3: 4 * math.pi * h ** 3 / 3
    }[dimension]

    return (1 - np.square(dist / h)) * (dimension + 2) / (c * 2)


def find_dist(coord1, coord2):
    if len(coord1) != len(coord2):
        raise ValueError('Coordinates must be of the same dimensions')

    return math.sqrt(sum(map(lambda x: (x[0] - x[1]) ** 2, zip(coord1, coord2))))


class FgdDetector:
    def __init__(self, filename):
        hs, hr, min_density = 8, 8, 100

        self.original_img = cv2.imread(filename)
        self.N = self.original_img.shape[0] * self.original_img.shape[1]
        self.segm_img, self.img_labels, self.num_labels = pymeanshift.segment(self.original_img,
                                                                              hs, hr, min_density)

        self.hs = math.sqrt(self.original_img.shape[0] ** 2 + self.original_img.shape[1] ** 2)
        self.hr = 300

        self.local_maxima = dict()
        self.contours = list()

        self.connections = None
        self.components = list()
        self._connected = list()

    def find_local_max(self):
        start = time.time()

        res = np.copy(self.original_img)

        boundary_labels = set(self.img_labels[0, :]) | set(self.img_labels[-1, :]) | \
                          set(self.img_labels[:, 0]) | set(self.img_labels[:, -1])

        for label in boundary_labels:
            res[np.where(self.img_labels == label)] = 0

        for ind in range(self.num_labels):
            if ind in boundary_labels: continue

            cluster_coords = np.where(self.img_labels == ind)

            cluster_color = self.segm_img[cluster_coords[0][0], cluster_coords[1][0]]
            cluster_size = len(cluster_coords[0])

            mask = np.zeros((self.original_img.shape[0], self.original_img.shape[1]), dtype=np.uint8)
            mask[cluster_coords] = 255

            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]

            self.contours.append((ind, cnt))

            M = cv2.moments(cnt)

            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            except ZeroDivisionError:
                continue

            leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
            rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
            topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
            bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

            radius = max(find_dist((cx, cy), leftmost), find_dist((cx, cy), rightmost),
                         find_dist((cx, cy), topmost), find_dist((cx, cy), bottommost))
            res[cy - 2: cy + 3, cx - 2: cx + 3] = [255, 0, 0]

            self.local_maxima[ind] = (cluster_size, cy, cx, RGB_to_Luv(*cluster_color), radius)

        print(time.time() - start)

    def remove_internal_clusters(self):
        start = time.time()

        for ind, contour in enumerate(self.contours):
            label, cnt = contour
            hull = cv2.convexHull(cnt)

            _maximas = self.local_maxima.copy()

            for ind, l_key in enumerate(_maximas.keys()):
                if (label != l_key and
                        cv2.pointPolygonTest(
                            hull,
                            (self.local_maxima[l_key][IND_SPATIAL + 1], self.local_maxima[l_key][IND_SPATIAL]),
                            False
                        ) > 0):
                    self.img_labels[np.where(self.img_labels == l_key)] = label

                    self.contours[ind:ind+1] = []
                    del self.local_maxima[l_key]

        print(time.time() - start)

    def get_neighbours(self):
        neighbours = {label: set() for label in self.local_maxima.keys()}

        for row in range(self.img_labels.shape[0] - 1):
            for col in range(self.img_labels.shape[1] - 1):
                    if self.img_labels[row, col] != self.img_labels[row + 1, col]:
                        if (self.img_labels[row, col] in self.local_maxima.keys() and
                                (self.img_labels[row + 1, col] in self.local_maxima.keys())):
                            neighbours[self.img_labels[row, col]].add(self.img_labels[row + 1, col])
                    if self.img_labels[row, col] != self.img_labels[row, col + 1]:
                        if (self.img_labels[row, col] in self.local_maxima.keys() and
                                (self.img_labels[row, col + 1] in self.local_maxima.keys())):
                            neighbours[self.img_labels[row, col]].add(self.img_labels[row, col + 1])

        return neighbours

    def filter_centroids(self):
        start = time.time()
        pdf_list = list()

        for label in self.local_maxima.keys():
            kernel_estimator = (
                lambda fs_element:
                    compute_kernel(
                        find_dist(self.local_maxima[label][IND_SPATIAL:IND_RANGE],
                                  self.local_maxima[fs_element][IND_SPATIAL:IND_RANGE]),
                        2, self.hs)
                    * compute_kernel(
                        find_dist(self.local_maxima[label][IND_RANGE],
                                  self.local_maxima[fs_element][IND_RANGE]),
                        3, self.hr)
            )

            _pdf = sum(map(kernel_estimator, self.local_maxima.keys()))
            _pdf /= (len(self.local_maxima) * self.hr ** 3 * self.hs ** 2)
            _pdf *= self.local_maxima[label][IND_SIZE]

            pdf_list.append((label, _pdf))

        pdf_avg = sum((pdf[1] for pdf in pdf_list))/len(pdf_list)

        print(time.time() - start)

        return [pdf[0] for pdf in pdf_list if pdf[1] > pdf_avg]

        # for pdf in pdf_list:
        #     if pdf[1] > pdf_avg:
        #         cv2.putText(
        #             pdf_map,
        #             '{:.3f}'.format(pdf[1] * 1e+14),
        #             pdf[0],
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.3,
        #             (255, 255, 255)
        #         )
        #
        #
        # dst = cv2.addWeighted(pdf_map, 0.4, self.segm_img, 0.6, 0)
        # cv2.imwrite('dst3.jpg', dst)

    def merge_cluster(self):
        centroids = self.filter_centroids()
        neighbours = self.get_neighbours()
        # print(centroids)
        # print(neighbours)

        edges = list()

        for centroid in centroids:
            color_ind1 = np.where(self.img_labels == centroid)
            _color1 = RGB_to_Luv(*self.original_img[color_ind1[0][0], color_ind1[1][0]])

            for bound_cluster in neighbours[centroid]:
                color_ind2 = np.where(self.img_labels == bound_cluster)
                _color2 = RGB_to_Luv(*self.original_img[color_ind2[0][0], color_ind2[1][0]])

                dist = (
                    find_dist(_color1, _color2) *
                    find_dist(self.local_maxima[centroid][IND_SPATIAL:IND_RANGE],
                              self.local_maxima[bound_cluster][IND_SPATIAL:IND_RANGE])
                )
                dist /= (self.hs * self.hr)

                if dist < 0.01:
                    edges.append((centroid, bound_cluster, dist))
                    print('{} to {}:\t{}'.format(centroid, bound_cluster, dist))

        self.connections = dict()

        for e1, e2, dist in edges:
            if e1 not in self.connections.keys(): self.connections[e1] = set()
            self.connections[e1].add(e2)
            if e2 not in self.connections.keys(): self.connections[e2] = set()
            self.connections[e2].add(e1)

        comp_ind = 0

        for connection in self.connections.keys():
            if connection in self._connected: continue

            self.components.append(list())
            self.components[comp_ind].append(connection)

            self._connected.append(connection)

            for deep_cnt in self.connections[connection]:
                self.deep_search(comp_ind, deep_cnt)
            comp_ind += 1

    def deep_search(self, comp_ind, search_key):
        if search_key in self._connected: return

        self._connected.append(search_key)
        self.components[comp_ind].append(search_key)

        for deep_cnt in self.connections[search_key]:
            self.deep_search(comp_ind, deep_cnt)


