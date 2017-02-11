import math
import numpy as np
import cv2
import time
import pymeanshift
from PIL import Image
import PIL


KERNEL_GAUSSIAN = 0
KERNEL_EPANECHNIKOV = 1


SPATIAL_DOMAIN = 0
RANGE_DOMAIN = 1

POST_DIM_NUM = 6
POST_IND_SIZE = 0
POST_IND_SPATIAL = 1
POST_IND_RANGE = 3


class FSProcessor:
    def __init__(self, domain_number, domain_type, domain_dim, domain_h, kernel=None):
        if len(domain_type) != domain_number or len(domain_dim) != domain_number or len(domain_h) != domain_number:
            raise ValueError("Length of dimensions container must be equal to domain_number")

        self.domain_number = domain_number
        self.domains = [dim for dim in zip(domain_type, domain_dim, domain_h)]

        self.kernel = FSProcessor.undefined
        self.kernel_type = FSProcessor.undefined
        if not kernel:
            self.define_kernel(KERNEL_EPANECHNIKOV)
        else:
            self.define_kernel(kernel)

        self.img = FSProcessor.undefined
        self.is_color = FSProcessor.undefined
        self.N = FSProcessor.undefined
        self.spatial_x = FSProcessor.undefined
        self.spatial_y = FSProcessor.undefined

        self.pdf = FSProcessor.undefined

    def find_local_max(self, filename):
        original_img = cv2.imread(filename)
        spatial_radius, range_radius, min_density = 8, 8, 100

        res = np.copy(original_img)
        segm_img, img_labels, num_labels = pymeanshift.segment(original_img, spatial_radius, range_radius, min_density)

        cv2.imwrite('segm1.jpg', segm_img)

        start = time.time()

        # Remove boundary clusters
        boundary_labels = set(img_labels[0, :]) | set(img_labels[-1, :]) | \
                          set(img_labels[:, 0]) | set(img_labels[:, -1])

        for label in boundary_labels:
            res[np.where(img_labels == label)] = 0

        local_maximum = np.empty((num_labels - len(boundary_labels), POST_DIM_NUM))

        cluster_index = 0
        for i in range(num_labels):
            if i in boundary_labels: continue

            cluster_coords = np.where(img_labels == i)

            cluster_color = segm_img[cluster_coords[0][0], cluster_coords[1][0]]
            cluster_size = len(cluster_coords[0])

            mask = np.zeros((original_img.shape[0], original_img.shape[1]), dtype=np.uint8)
            mask[cluster_coords] = 255

            image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            M = cv2.moments(contours[0])

            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            except ZeroDivisionError:
                continue

            res[cy - 2: cy + 3, cx - 2: cx + 3] = [255, 0, 0]

            local_maximum[cluster_index][POST_IND_SIZE:POST_IND_SPATIAL] = cluster_size
            local_maximum[cluster_index][POST_IND_SPATIAL:POST_IND_RANGE] = (cy, cx)
            local_maximum[cluster_index][POST_IND_RANGE:] = FSProcessor.RGB_to_Luv(*cluster_color)

            print(local_maximum[cluster_index])

            cluster_index += 1

        print(time.time() - start)

        cv2.imwrite('lm1.jpg', res)

    @staticmethod
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

    def define_kernel(self, kernel_type):
        if kernel_type == KERNEL_GAUSSIAN:

            def gaussian(dist, dimensions):
                return np.exp(np.square(dist) * -0.5) * pow(math.pi * 2, dimensions / -2)

            self.kernel = lambda dist, dimension, h: gaussian(dist, dimension)
            self.kernel_type = KERNEL_GAUSSIAN
        elif kernel_type == KERNEL_EPANECHNIKOV:
            def epanechnikov(dist, dimension, h):
                c = 1
                if dimension == 1:
                    c = 2 * math.pi * h
                elif dimension == 2:
                    c = math.pi * h ** 2
                elif dimension == 3:
                    c = 4 * math.pi * h ** 3 / 3
                else:
                    # TODO custom c
                    pass

                return (1 - np.square(dist)) * (dimension + 2) / (c * 2)

            self.kernel = lambda dist, dimension, h: epanechnikov(dist, dimension, h)
            self.kernel_type = KERNEL_EPANECHNIKOV
        else:
            raise ValueError("Kernel type is not understood")

    def find_dist(self, domain_type, center_matrix):
        if domain_type == SPATIAL_DOMAIN:
            return np.sqrt(
                np.square(center_matrix[0] - self.spatial_x)
                + np.square(center_matrix[1] - self.spatial_y)
            )

        elif domain_type == RANGE_DOMAIN:
            if self.is_color:
                return np.sqrt(
                    np.square(center_matrix[0] - self.img[:, :, 0])
                    + np.square(center_matrix[1] - self.img[:, :, 1])
                    + np.square(center_matrix[2] - self.img[:, :, 2])
                )
            else:
                return np.abs(center_matrix - self.img)

    def compute_pdf(self):
        print(self.N)
        self.pdf = np.empty((self.img.shape[0], self.img.shape[1]), dtype=np.float32)

        x_matrix = np.zeros((self.img.shape[0], self.img.shape[1]))
        y_matrix = np.zeros((self.img.shape[0], self.img.shape[1]))

        for y in range(self.img.shape[0]):
            start = time.time()
            for x in range(self.img.shape[1]):
                composite_kernels = np.ones((self.img.shape[0], self.img.shape[1]), dtype=np.float64)

                for _type, dim, h in self.domains:
                    if _type == SPATIAL_DOMAIN:
                        dist = self.find_dist(_type, (x_matrix, y_matrix))
                    elif _type == RANGE_DOMAIN:
                        if self.is_color:
                            L_channel = np.empty((self.img.shape[0], self.img.shape[1]))
                            u_channel = np.empty((self.img.shape[0], self.img.shape[1]))
                            v_channel = np.empty((self.img.shape[0], self.img.shape[1]))

                            L_channel.fill(self.img[y, x, 0])
                            u_channel.fill(self.img[y, x, 1])
                            v_channel.fill(self.img[y, x, 2])

                            dist = self.find_dist(_type, (L_channel, u_channel, v_channel))
                        else:
                            dist = self.find_dist(_type,
                                                  np.empty((self.img.shape[0], self.img.shape[1])).fill(self.img[y,x]))

                    if self.kernel_type == KERNEL_EPANECHNIKOV:
                        dist[np.where(dist >= h)] = h

                    composite_kernels *= self.kernel(dist / h, dim, h)
                    composite_kernels /= h ** dim

                self.pdf[y, x] = np.sum(composite_kernels) / self.N

                x_matrix += 1
            print(y, time.time() - start)
            y_matrix += 1
            x_matrix = np.zeros((self.img.shape[0], self.img.shape[1]))

    def write_results(self, filename):
        res = open(filename, 'w')
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                res.write('{0},{1},{2}\n'.format(x, y, self.pdf[y, x] * 100000000000000))

    def load_img(self, filename):
        self.img = np.float32(cv2.imread(filename))
        self.img *= 1./255

        if len(self.img.shape) == 3:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)
        else:
            self.is_color = False

        self.N = self.img.shape[0] * self.img.shape[1]

        self.spatial_x = np.tile(np.arange(self.img.shape[1]), (self.img.shape[0], 1))
        self.spatial_y = np.repeat(np.arange(self.img.shape[0]).reshape((1,self.img.shape[0])),
                                   self.img.shape[1], axis=1).reshape((self.img.shape[0], self.img.shape[1]))

    @staticmethod
    def get_undefined_object():
        raise KeyError("FSProcessor: trying to get undefined property")

    undefined = property(get_undefined_object)
