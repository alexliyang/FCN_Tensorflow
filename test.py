import scipy.io

if __name__ == "__main__":
    data = scipy.io.loadmat('./imagenet-vgg-verydeep-19.mat')
    print(data['layers'])
    print(7*7*256)