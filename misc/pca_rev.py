from sklearn import datasets
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

digits_data = datasets.load_digits()
n = len(digits_data.images)

# flatten array as input to PCA
image_data = digits_data.images.reshape((n,-1))
print(image_data.shape)

labels = digits_data.target

# Fit PCA transformer
pca_transformer = PCA(n_components=0.8)
pca_images = pca_transformer.fit_transform(image_data)

print(pca_transformer.explained_variance_ratio_[:3].sum())

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
for i in range(100):
    ax.scatter(pca_images[i,0],pca_images[i,1],pca_images[i,2],
                marker=r'${}$'.format(labels[i]),s=64)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
plt.show()