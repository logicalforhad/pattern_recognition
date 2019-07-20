import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def projection_function(X, eigenset):
    return np.dot(X, eigenset.T)


def PCA(n_biggest_vector):
    # calculate the overall mean of data
    overall_data_mean = np.mean(data_x, axis=0)
    # normalize to zero mean
    zero_mean_data = data_x - overall_data_mean
    # transpose data
    covariance_matrix = np.cov(np.transpose(zero_mean_data))
    # derive eig values and vectors
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    # return n-biggest eigen-vectors
    return eigen_vectors.T[::-1][:n_biggest_vector]


def LDA(n_biggest_vector):
    # calculating the overall mean of data
    overall_data_mean = np.mean(data_x, axis=0)
    # class_means contains the means related for each class
    classes = np.unique(y_label)

    n_features = data_x.shape[1]  # 500
    n_classes = len(classes)

    class_mean = np.zeros((n_classes, n_features))
    covariances = np.zeros((n_classes, n_features, n_features))

    for i in range(n_classes):
        indices = np.where(y_label == i + 1)[0]
        class_mean[i, :] = np.mean(data_x[indices, :], axis=0)
        covariances[i, :, :] = np.cov(data_x[indices, :].T)

    S_w = np.zeros((n_features, n_features))

    for i in range(n_classes):
        S_w = S_w + covariances[i, :, :]

    S_b = np.zeros((n_features, n_features))

    for i in range(n_classes):
        S_b = S_b + np.outer(class_mean[i, :] - overall_data_mean, class_mean[i, :] - overall_data_mean)

    C = np.dot(np.linalg.pinv(S_w), S_b)
    eigen_values, eigen_vectors = np.linalg.eigh(C)
    return eigen_vectors.T[::-1][:n_biggest_vector]


if __name__ == "__main__":

    data_x = np.loadtxt("data-dimred-X.csv", dtype=np.float, delimiter=',').T
    y_label = np.loadtxt("data-dimred-y.csv", dtype=np.int, delimiter='\n')

    # PCA 2D drawing
    final_pca = projection_function(data_x, PCA(2))
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(final_pca.T[0][np.where(y_label == i)],
                    final_pca.T[1][np.where(y_label == i)], color=j)
    plt.title('PCA Algorithm 2D')
    plt.legend(['#1', '#2', '#3'], loc=3)
    plt.savefig('Figure_PCA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # LDA 2D drawing
    final_lda = projection_function(data_x, LDA(2))
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(final_lda.T[0][np.where(y_label == i)],
                    final_lda.T[1][np.where(y_label == i)], color=j)
    plt.title('LDA Algorithm 2D')
    plt.legend(['#1', '#2', '#3'], loc=3)
    plt.savefig('Figure_LDA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # PCA 3D drawing
    final_pca_3d = projection_function(data_x, PCA(3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D(final_pca_3d.T[0][np.where(y_label == i)],
                      final_pca_3d.T[1][np.where(y_label == i)],
                      final_pca_3d.T[2][np.where(y_label == i)], color=j)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('PCA Algorithm 3D')
    plt.savefig('Figure_PCA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # LDA 3D drawing
    final_lda_3d = projection_function(data_x, LDA(3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D(final_lda_3d.T[0][np.where(y_label == i)],
                      final_lda_3d.T[1][np.where(y_label == i)],
                      final_lda_3d.T[2][np.where(y_label == i)], color=j)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('LDA Algorithm 3D')
    plt.savefig('Figure_LDA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()