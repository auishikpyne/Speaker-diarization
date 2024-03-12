import glob
from tqdm import tqdm
import numpy as np
import nemo.collections.asr as nemo_asr
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

speaker_model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained("nvidia/speakerverification_en_titanet_large")

# print(speaker_model)

audio_files = glob.glob("/UPDS/TTS/deliverables2/create/*.flac")

embeddings = []

for file in tqdm(audio_files):
    emb = speaker_model.get_embedding(file)
    
    emb = emb.cpu().detach().numpy()  # Transfer to CPU and convert to NumPy array
    emb = emb.squeeze()
    
    embeddings.append(emb)
    
embeddings = np.array(embeddings)

tsne = TSNE(n_components=3, random_state=0)
embedded = tsne.fit_transform(embeddings)

np.savetxt('tsne_embedded_values3d.txt', embedded)

cluster_range = range(8, 18)
silhouette_scores = []

for n_clusters in tqdm(cluster_range):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(embedded)  # Apply clustering to the t-SNE data
    silhouette_avg = silhouette_score(embedded, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot the silhouette scores for different cluster numbers
plt.plot(cluster_range, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different Cluster Numbers')
plt.savefig("elbow_plot_tsne3d.png")
plt.show()


# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# file_path = '/home/auishik/nvidia_nemo/NeMo/tutorials/speaker_tasks/tsne_embedded_values.txt'  

# # Open the file in read mode
# with open(file_path, 'r') as file:
#     # Read the contents of the file into a variable
#     lines = file.read().splitlines()

# # Convert the lines into a NumPy array (assuming space-separated values in each line)
# tsne_embedded = np.array([list(map(float, line.split())) for line in lines])

# # Apply K-means clustering with the chosen K
# kmeans = KMeans(n_clusters=14, random_state=0)
# cluster_labels = kmeans.fit_predict(tsne_embedded)

# tsne_embedding = tsne_embedded  # Replace with your t-SNE embedding

# from mpl_toolkits.mplot3d import Axes3D

# # Create a 3D scatter plot for visualization
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot points in 3D, color-coded by cluster labels
# scatter = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], tsne_embedding[:, 2], c=cluster_labels, cmap='viridis')

# # Add labels and title
# ax.set_xlabel('Dimension 1')
# ax.set_ylabel('Dimension 2')
# ax.set_zlabel('Dimension 3')
# plt.title(f'K-means Clustering (K={14})')

# # Add a colorbar to the right of the plot
# cbar = plt.colorbar(scatter)
# cbar.set_label('Cluster Labels')

# plt.savefig("kmeans_plot.png")
# # Show the 3D scatter plot
# plt.show()