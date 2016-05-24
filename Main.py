__author__ = "andreadelfante"

from __init__ import *

def main():
    urlMap = UrlMap(file_path="urlMap.txt")
    groundTruth = GroundTruth(fpath="groundTruth.txt")

if __name__ == '__main__':
    main()