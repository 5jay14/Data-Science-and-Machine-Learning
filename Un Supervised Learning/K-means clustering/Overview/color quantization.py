import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# mmatplotlib. image reads the images(.jpg, .png) and converts into numpy arrays

image_as_array = mpimg.imread(r"C:\Users\vijay\Desktop\DS ML\DATA - Copy\palm_trees.jpg")
print(image_as_array)  # H, W, C for each pixel

plt.imshow(image_as_array)  # plt.imshow needs a numpy array, it will not read the image directly from the path
plt.show()

# reshaping 3d into 2d
# (H, W, C) ==> (H*W, C)

(h, w, c) = image_as_array.shape
image_as_2d_array = image_as_array.reshape(h * w, c)
print(len(image_as_2d_array.shape)) # 2d

# conversion is done because, K means accepts features that are 2 dimensions

from sklearn.cluster import  KMeans

model = KMeans(n_clusters=6)
labels = model.fit_predict(image_as_2d_array)
print(labels)
rgb_codes = model.cluster_centers_.round(0).astype(int)# averaged out colors

print(rgb_codes)