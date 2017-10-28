import tifffile as tiff
import matplotlib.pyplot

if __name__ == "__main__":
    im_2015 = tiff.imread("./2015.tif").transpose([1, 2, 0])
    print("the shape is ", im_2015.shape)
    tiff.imshow(im_2015)