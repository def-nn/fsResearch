from FSProcessor import FSProcessor, SPATIAL_DOMAIN, RANGE_DOMAIN


if __name__ == '__main__':
    processor = FSProcessor(2, (SPATIAL_DOMAIN, RANGE_DOMAIN), (2, 3), (8, 8))
    processor.load_img('../../../Python/cv/images/logo.jpg')

    processor.compute_pdf()
    processor.write_results()