import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm import tqdm
np.random.seed(5)
import sys
sys.path.insert(0, 'D:/Dossier_These_code_and_datas/LDA_all/function_LDA_package')
from function_used_in_LDA_method import continuous_data_files_concat_kmeans_reattached_V2,continuous_data_files_concat_kmeans_reattached,LDA_function, generate_data, storage_position, relabeling, relabeling_hedi_style_V2, labeling_datas_Kmeans_KNN,LDA_function_makeblop_form, Generate_patient_with_make_blobs


#Fix Directory
patient_frame_to_store = 'Patient_generated_binary_only_mutliple_gaussian'

#Chose to work on continuous data
figures_storage ,frame_storage = 'Figures_LDA_on_binary_data_generated' , 'Data_frame_LDA_on_binary_data_generated'
path_to_store_figures,path_to_store_frame = storage_position(patient_frame_to_store,figures_storage ,frame_storage, abspath =  os.path.abspath(__file__) )


#Generate patients characteristics
number_of_patient = 50
number_of_cell = 10000

vectors_of_probability = [[0.9,0.1,0.3],[0.1,0.7,0.6],[0.1,0.9,0.3]]


#Clustering possibility
number_of_cluster_k_means = 30 #Number_of_cluster_for_k_means

#LDA characteristics
diviser_of_matrix = 100
number_of_calculation = 200
number_of_topic_to_test = 4
cutting_tree = 0.4

#Lets Generate Patients and store them
phenotype_code = Generate_patient_with_make_blobs(number_of_patient,number_of_cell,vectors_of_probability,patient_frame_to_store, return_combination = True)

#Now we cluster cells from different patients
full_dataframe , dataframe_for_LDA_to_use = continuous_data_files_concat_kmeans_reattached_V2(path_to_store_frame = patient_frame_to_store,number_of_cluster = number_of_cluster_k_means, number_of_cell = number_of_cell)
real_phenotype_code = phenotype_code['Phenotype_Number']

#Generate LDA
accuracy_score_store = []
for runrun in tqdm(range(2,number_of_topic_to_test +1)):
    norm_theta = LDA_function_makeblop_form(dataframe_for_LDA = dataframe_for_LDA_to_use,diviser_of_matrix = diviser_of_matrix,runrun = runrun,number_of_calculation = number_of_calculation, alpha= 1/number_of_calculation, beta=1/number_of_calculation)
    
    linked = linkage(norm_theta, method='complete', metric='euclidean', optimal_ordering = True)
    labelList = dataframe_for_LDA_to_use.T.columns.to_list()
    print(norm_theta)

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.savefig(path_to_store_figures + f'/Cluster_dendrogram_for_topic_%s.png'%runrun)
    plt.close()

    hierarchical_result = fcluster(linked, cutting_tree*norm_theta.max(),'distance')
    frame_binary_with_threshold = pd.DataFrame(norm_theta,index=labelList,columns= [f'Topic {i}' for i in range(runrun)])

    frame_binary_with_threshold['Statut_number'] =  frame_binary_with_threshold.index
    frame_binary_with_threshold.reset_index(drop=True, inplace=True)
    frame_binary_with_threshold['Hierarchical_clustering'] = hierarchical_result
    frame_binary_with_threshold['Phenotype'] = real_phenotype_code
    frame_for_accuracy = frame_binary_with_threshold.groupby(['Phenotype','Hierarchical_clustering']).size().unstack(fill_value=0)
    print(frame_for_accuracy)
    calc_of_accuracy = accuracy_score(frame_binary_with_threshold['Phenotype'], frame_binary_with_threshold['Hierarchical_clustering'])
    print(f'Accuracy = {calc_of_accuracy}')

