import math
import numpy as np
import cv2
import time


KERNEL_GAUSSIAN = 0
KERNEL_EPANECHNIKOV = 1


SPATIAL_DOMAIN = 0
RANGE_DOMAIN = 1


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
            y_matrix += 1
            x_matrix = np.zeros((self.img.shape[0], self.img.shape[1]))

    def write_results(self):
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                print(y, x, self.pdf[y, x])

    def load_img(self, file):
        self.img = np.float32(cv2.imread(file))
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
