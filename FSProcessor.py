import math
import numpy as np
import cv2


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
        if not kernel:
            self.define_kernel(KERNEL_EPANECHNIKOV)
        else:
            self.define_kernel(kernel)

        self.img = FSProcessor.undefined

    def define_kernel(self, kernel_type):
        if kernel_type == KERNEL_GAUSSIAN:

            def gaussian(dist, dimensions, h):
                return pow(2 * math.pi, dimensions / -2) * math.exp(-0.5 * dist ** 2)

            self.kernel = lambda dist, dimension: gaussian(dist, dimension)
        elif kernel_type == KERNEL_EPANECHNIKOV:

            def epanechnikov(dist, dimension, h):
                if dist < 1:
                    # TODO norm const c - volume of the unit d-dimensional sphere
                    return (dimension + 2) * (1 - dist ** 2) / 2
                else: return 0

            self.kernel = lambda dist, dimension: epanechnikov(dist, dimension)
        else:
            raise ValueError("Kernel type is not understood")

    def load_img(self, file):
        self.img = np.float32(cv2.imread(file))
        self.img *= 1./255
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2Luv)

    @staticmethod
    def get_undefined_object():
        raise KeyError("FSProcessor: trying to get undefined property")

    undefined = property(get_undefined_object)
