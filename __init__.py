import numpy as np
import cv2
from ForegroundDetector import FgdDetector


if __name__ == '__main__':

    fgDetector = FgdDetector('/Users/Anastasia/Python/cv/images/food1.jpg')
    fgDetector.find_local_max()
    fgDetector.remove_internal_clusters()
    fgDetector.merge_cluster()

    # _fyle = open('/Users/Anastasia/Documents/work/fsResearch/pdf_3.txt')
    # rows = list()
    # cols = list()
    # pdf_set = list()
    # for line in _fyle:
    #     row = col = pdf = ''
    #     exit_flag = 0
    #     for sym in line:
    #         if sym.isdigit():
    #             row += sym
    #             exit_flag += 1
    #         else: break
    #     exit_flag += 1
    #     for sym in line[exit_flag :]:
    #         if sym.isdigit():
    #             col += sym
    #             exit_flag += 1
    #         else: break
    #     exit_flag += 1
    #     pdf = line[exit_flag:]
    #     print(pdf)
    #
    #     row, col = int(row), int(col)
    #
    #
    #     rows.append(row)
    #     cols.append(col)
    #     pdf_set.append(pdf)
    # _fyle.close()
    #
    # _file_new = open('/Users/Anastasia/Documents/work/fsResearch/pdf_3.txt', 'w')
    # for row, col, pdf in zip(rows, cols, pdf_set):
    #     _file_new.write("{}\t{}\t{}\n".format(row, col, pdf * 1e+3))
    #
    # _file_new.close()


