import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import time
from discord_webhook import DiscordWebhook
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster
from scipy.cluster.hierarchy import dendrogram, linkage
from tqdm import tqdm
np.random.seed(5)
import sys
from datetime import timedelta
#sys.path.insert(0, '~/LDA_Project/function_LDA_package')
from function_used_in_LDA_method import continuous_data_files_concat_kmeans_reattached_V2,continuous_data_files_concat_kmeans_reattached,LDA_function, generate_data, storage_position, relabeling, relabeling_hedi_style_V2, labeling_datas_Kmeans_KNN,LDA_function_makeblop_form, Generate_patient_with_make_blobs, number_of_phenotypes_to_test_with_vector
start_time = time.monotonic()




#Fix Directory
patient_frame_to_store = 'Patient_generated_binary_only_mutliple_gaussian'

#Chose to work on continuous data
figures_storage ,frame_storage = 'Figures_LDA_on_binary_data_generated' , 'Data_frame_LDA_on_binary_data_generated'
path_to_store_figures,path_to_store_frame = storage_position(patient_frame_to_store,figures_storage ,frame_storage, abspath =  os.path.abspath(__file__) )

#Generate patients characteristics
number_of_patient = 1000
number_of_cell = 10000

dimension_to_test = [2,3,4,5] #number of dimension
for dimension in dimension_to_test:

    vectors_of_probability = number_of_phenotypes_to_test_with_vector(dimension)

    #Clustering possibility
    number_of_cluster_k_means = 40 #Number_of_cluster_for_k_means

    #LDA characteristics
    diviser_of_matrix = 100
    number_of_calculation = 200
    cutting_tree = 1

    #Lets Generate Patients and store them
    phenotype_code = Generate_patient_with_make_blobs(number_of_patient,number_of_cell,vectors_of_probability,patient_frame_to_store, return_combination = True)

    #Now we cluster cells from different patients
    full_dataframe , dataframe_for_LDA_to_use = continuous_data_files_concat_kmeans_reattached_V2(path_to_store_frame = patient_frame_to_store,number_of_cluster = number_of_cluster_k_means, number_of_cell = number_of_cell)
    phenotype_code_frame = pd.DataFrame(phenotype_code)

    #Generate LDA
    #runrun = len(vectors_of_probability) #if you want to use len of vector = number of cluster
    runrun = 3 #if you make the assuption that the number of group can't be more than 4
    accuracy_score_store = []

    norm_theta = LDA_function_makeblop_form(dataframe_for_LDA = dataframe_for_LDA_to_use,diviser_of_matrix = diviser_of_matrix,runrun = runrun,number_of_calculation = number_of_calculation, alpha= 1/number_of_calculation, beta=1/number_of_calculation)
    pd.DataFrame(norm_theta).to_csv(path_to_store_frame + f'/Norm_Theta_frame_{runrun}_{len(vectors_of_probability)}_phenotypes.csv')
    linked = linkage(norm_theta, method='complete', metric='euclidean', optimal_ordering = True)
    labelList = dataframe_for_LDA_to_use.T.columns.to_list()
    print(norm_theta)


    plt.figure(figsize=(10, 7))
    dendrogram(linked,
                orientation='top',
                labels=labelList,
                distance_sort='descending',
                show_leaf_counts=True)
    plt.savefig(path_to_store_figures + f'/Cluster_dendrogram_for_topic_{runrun}_{len(vectors_of_probability)}_phenotypes_.png')
    plt.close()

    hierarchical_result = fcluster(linked, cutting_tree*norm_theta.max(),'distance')
    frame_binary_with_threshold = pd.DataFrame(norm_theta,index=labelList,columns= [f'Topic {i}' for i in range(runrun)])

    frame_binary_with_threshold['Patient_Number'] =  frame_binary_with_threshold.index
    frame_binary_with_threshold.reset_index(drop=True, inplace=True)
    frame_binary_with_threshold['Hierarchical_clustering'] = hierarchical_result

    global_frame = pd.merge(frame_binary_with_threshold, phenotype_code_frame, on="Patient_Number")
    global_frame.to_csv(path_to_store_frame + f'/Global_Dataframe_{runrun}_{len(vectors_of_probability)}_phenotypes.csv')

    frame_for_accuracy = global_frame.groupby(['Phenotype_Number','Hierarchical_clustering']).size().unstack(fill_value=0)
    frame_for_accuracy.to_csv(path_to_store_frame + f'/Dataframe_for_accuracy_{runrun}_{len(vectors_of_probability)}_phenotypes.csv')
    print(frame_for_accuracy)

    calc_of_accuracy = accuracy_score(global_frame['Phenotype_Number'], global_frame['Hierarchical_clustering'])
    print(f'Accuracy = {calc_of_accuracy}')



    #Ping job done
    #webhook = DiscordWebhook(url = '' ,content=f"@here \n\
    #    **Your File: **{os.path.basename(__file__)} \n\
    #    **Execution:** succes :white_check_mark: for {dimension}\n\
    #    **Duration Time:** {timedelta(seconds = time.monotonic()-start_time)}" )
    #webhook.execute()
