## PCA

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#1 - Scale data!!
scaler = StandardScaler()
scaler.fit(df)

scaled_data = scaler.transform(df)

pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')

pca.components_