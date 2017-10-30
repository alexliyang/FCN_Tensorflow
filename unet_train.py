from tf_unet import unet, util, image_util
#preparing data loading
data_provider = image_util.ImageDataProvider("../training_pic/*.tif")

#setup & training
net = unet.Unet(layers=3, features_root=64, channels=3, n_class=2)
trainer = unet.Trainer(net)
path = trainer.train(data_provider, "./unet", training_iters=32, epochs=100)