import time
from FSProcessor import FSProcessor, SPATIAL_DOMAIN, RANGE_DOMAIN


if __name__ == '__main__':
    processor = FSProcessor(2, (SPATIAL_DOMAIN, RANGE_DOMAIN), (2, 3), (8, 8))
    processor.load_img('../../../Python/cv/images/woman_face.jpg')

    start = time.time()
    processor.compute_pdf()
    t1 = time.time()
    print(t1 - start)
    processor.write_results('/Users/Anastasia/Documents/work/pdf_face.txt')
    print(time.time() - t1)
