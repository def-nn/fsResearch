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
        self.N = FSProcessor.undefined
        self.pdf = FSProcessor.undefined

    def define_kernel(self, kernel_type):
        if kernel_type == KERNEL_GAUSSIAN:

            def gaussian(dist, dimensions):
                return pow(2 * math.pi, dimensions / -2) * math.exp(-0.5 * dist ** 2)

            self.kernel = lambda dist, dimension, h: gaussian(dist, dimension)
            self.kernel_type = KERNEL_GAUSSIAN
        elif kernel_type == KERNEL_EPANECHNIKOV:
            # Called only when dist < 1
            def epanechnikov(dist, dimension, h):
                c = 1
                if dimension == 2:
                    c = math.pi * h ** 2
                elif dimension == 3:
                    c = 4 * math.pi * h ** 3 / 3
                else:
                    # TODO custom c
                    pass

                return (dimension + 2) * (1 - dist ** 2) / (2 * c)

            self.kernel = lambda dist, dimension, h: epanechnikov(dist, dimension, h)
            self.kernel_type = KERNEL_EPANECHNIKOV
        else:
            raise ValueError("Kernel type is not understood")

    def find_dist(self, coord1, coord2):
        dist = 0
        for axis1, axis2 in zip(coord1, coord2):
            dist += (axis1 - axis2) ** 2
        return math.sqrt(dist)

    def compute_kernel(self, coord1, coord2, domain_dim, domain_h):
        kdf = 0
        dist = self.find_dist(coord1, coord2)

        if self.kernel_type == KERNEL_EPANECHNIKOV:
            if dist < domain_h: kdf = self.kernel(dist / domain_h, domain_dim, domain_h)
        else:
            kdf = self.kernel(dist / domain_h, domain_dim, domain_h)

        return kdf

    def compute_pdf(self):
        print(self.N)
        self.pdf = np.empty((self.img.shape[0], self.img.shape[1]), dtype=np.float32)

        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                start = time.time()
                # TODO estimate normalized constant that makes K(x) integrate to one
                kernel_sum = 0

                for _y in range(self.img.shape[0]):
                    for _x in range(self.img.shape[1]):
                        composite_kernel = 1

                        for _type, dim, h in self.domains:
                            coord1, coord2 = {
                                SPATIAL_DOMAIN: ((y,x), (_y, _x)),
                                RANGE_DOMAIN: (self.img[y,x], self.img[_y, _x])
                            }[_type]

                            composite_kernel *= self.compute_kernel(coord1, coord2, dim, h)
                            composite_kernel /= h ** dim

                        kernel_sum += composite_kernel

                self.pdf[y,x] = kernel_sum / self.N
                print(time.time() - start)

    def write_results(self):
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                print(y, x, self.pdf[y, x])

    def load_img(self, file):
        self.img = np.float32(cv2.imread(file))
        self.img *= 1./255
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)

        self.N = self.img.shape[0] * self.img.shape[1]

    @staticmethod
    def get_undefined_object():
        raise KeyError("FSProcessor: trying to get undefined property")

    undefined = property(get_undefined_object)
