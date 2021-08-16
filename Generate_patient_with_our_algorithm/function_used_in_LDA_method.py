import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.ma import count
import pandas as pd
from plotnine.labels import labs
from tqdm import tqdm
import os
import random
from glob import glob 
from scipy import stats
from scipy.spatial import cKDTree as KDTree
from sklearn.cluster import KMeans
from collections import Counter
import time
import itertools
from sklearn.datasets import make_blobs

np.random.seed(5)



def storage_position(patient_frame_to_store,figures_storage,frame_storage, abspath):
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    path = os.getcwd()
    PATH_storage = path
    print(PATH_storage)

    patient_frame_to_store = 'Patient_generated_binary_only_mutliple_gaussian'
    if not os.path.exists(patient_frame_to_store):
        os.makedirs(patient_frame_to_store)
    path_to_store_patient_frame = os.path.join(PATH_storage, patient_frame_to_store)

    dname = os.path.dirname(abspath)
    os.chdir(dname)

    path = os.getcwd()
    PATH_storage = path
    print(PATH_storage)

    if not os.path.exists(figures_storage):
        os.makedirs(figures_storage)
    path_to_store_figures = os.path.join(PATH_storage, figures_storage)

    if not os.path.exists(frame_storage):
        os.makedirs(frame_storage)
    path_to_store_frame = os.path.join(PATH_storage, frame_storage)

    return path_to_store_figures,path_to_store_frame


def generate_data(number_of_cell_to_generate, proba_tupple):
    size_list = number_of_cell_to_generate
    probas = proba_tupple
    number_of_iteration = tuple([int(size_list*x) for x in probas])

    list_of_iteration = []
    for i in range(len(number_of_iteration)):
        a = [i]*number_of_iteration[i]
        list_of_iteration += a

    return list_of_iteration


def number_of_phenotypes_to_test_with_vector(dimension):
    list_vecteur = np.linspace(0.1,1,10)
    list_of_vector_of_probability = []
    for element in itertools.product(np.round(list_vecteur,3), repeat = dimension):
        if round(sum(element),3) == 1:
            list_of_vector_of_probability.append(element)
    list_of_vector_of_probability = [list(ele) for ele in list_of_vector_of_probability]

    print(f'Number of possible phenotype combination for {dimension} dimension is {len(list_of_vector_of_probability)} phenotypes')
    return list_of_vector_of_probability


def Generate_patient_with_make_blobs(number_of_patient,number_of_cell,vectors_of_probability, patient_frame_to_store, return_combination):
    for i,x in enumerate(vectors_of_probability):
        globals()[f"vector_of_probability_{i}"] = x

    splitted_vector = np.round(np.linspace(0, number_of_patient, len(vectors_of_probability)+1)[1:]).astype(int)
    list_of_array_patients = np.split(range(number_of_patient), splitted_vector)[:-1]


    patient_and_phenotype = {'Patient_Number':[] ,'Phenotype_Number':[], 'Patient_combination_phenotype':[]}
    first_step = 0


    for group_patient_phenotype , list_patient in tqdm(enumerate(list_of_array_patients)):
        vector_of_probability = globals()[f"vector_of_probability_{group_patient_phenotype}"]
        number_of_dimension = len(vector_of_probability)

        if first_step == 0:
            for number in range(number_of_dimension):
                #value = 0.1
                globals()[f"cluster_std_{number}"] = random.uniform(0, 1)
                #value +=  0.1
            for number in range(number_of_dimension):
                advance = 42
                globals()[f"random_state{number}"] =  advance
                advance += random.uniform(1, 15)
            first_step += 1

        for patient in tqdm(list_patient):
            phenotype_continuous_data = []
            phenotype_binary_data = []
            number_of_feature_to_generate = 1


            #Patient Phenotype
            patient_and_phenotype['Patient_Number'].append(f'patient_N°{str(patient).zfill(3)}')
            patient_and_phenotype['Phenotype_Number'].append(group_patient_phenotype+1)
            patient_and_phenotype['Patient_combination_phenotype'].append(vector_of_probability)

            #First define random state and dimentionality
            # for number in range(number_of_dimension):
            #     globals()[f"random_state{number}"] =  random.randint(0, 100)

            for i in range(len(vector_of_probability)):
                globals()[f"cell_distribution_col_{i}"] = np.round([vector_of_probability[i], 1- vector_of_probability[i]],3)


            for i in range(number_of_dimension):
                #generate distribution
                distribution = 'cell_distribution_col_'+f'{i}' #integer for the distribution
                cells_distributed = (np.array(globals()[distribution]) * number_of_cell).astype(int) #distribution

                X, y = make_blobs(n_samples=cells_distributed, centers=None, n_features= number_of_feature_to_generate, cluster_std=globals()[f"cluster_std_{i}"], random_state= globals()[f"random_state{i}"])
                phenotype_continuous_data.append(X)
                phenotype_binary_data.append(y)
                #globals()[f"random_state{i}"] += 10

            phenotype_continuous_data_patient = pd.DataFrame(list(map(np.ravel,phenotype_continuous_data))).T
            phenotype_binary_data_patient = pd.DataFrame(list(map(np.ravel,phenotype_binary_data))).T

            phenotype_continuous_data_patient.to_csv(patient_frame_to_store +f'/Generated_file_for_patient_N°{str(patient).zfill(3)}.csv')
            phenotype_binary_data_patient.to_csv(patient_frame_to_store + f'/Generated_binary_file_for_patient_N°{str(patient).zfill(3)}.csv')
            

    pd.DataFrame(patient_and_phenotype).to_csv(patient_frame_to_store + f'/Patient_number_and_phenotype_code.csv')


    if return_combination == True:
        return patient_and_phenotype







def LDA_function(dataframe_for_LDA,diviser_of_matrix,runrun,number_of_calculation, alpha, beta):
    excluded_patient = 0
    vocabulary  =  list(range(dataframe_for_LDA.shape[1]))
    raw_data_T4 = dataframe_for_LDA.T
    data_T4 = raw_data_T4.T
    t4_ = data_T4.to_numpy()/diviser_of_matrix

    #t4_ = np.around(t4_)
    docs = []
    npatients, nvocabulary = t4_.shape
    for n in range (npatients):
        current_doc = []
        doc = t4_[n,:]
        for i in range(nvocabulary):
            for _ in range(int(doc[i])):
                current_doc.append(i)
        docs.append(current_doc)
                
            

    D = len(docs)        # number of documents
    V = len(vocabulary)  # size of the vocabulary 
    T = runrun            # number of topics

    # the parameter of the Dirichlet prior on the per-document topic distributions  #Faire varier
    # the parameter of the Dirichlet prior on the per-topic word distribution


    z_d_n = [[0 for _ in range(len(d))] for d in docs]  # z_i_j
    theta_d_z = np.zeros((D, T))
    phi_z_w = np.zeros((T, V))
    n_d = np.zeros((D))
    n_z = np.zeros((T))

    ## Initialize the parameters
    # m: doc id
    for d, doc in enumerate(docs):  
        # n: id of word inside document, w: id of the word globally
        for n, w in enumerate(doc):
            # assign a topic randomly to words
            z_d_n[d][n] = int(np.random.randint(T))
            # get the topic for word n in document m
            z = z_d_n[d][n]
            # keep track of our counts
            theta_d_z[d][z] += 1
            phi_z_w[z, w] += 1
            n_z[z] += 1
            n_d[d] += 1

    #for iteration in tqdm(range(number_of_calculation)):
    for iteration in range(number_of_calculation):
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                # get the topic for word n in document m
                z = z_d_n[d][n]

                # decrement counts for word w with associated topic z
                theta_d_z[d][z] -= 1
                phi_z_w[z, w] -= 1
                n_z[z] -= 1

                # sample new topic from a multinomial according to our formular
                p_d_t = (theta_d_z[d] + alpha) / (n_d[d] - 1 + T * alpha)
                p_t_w = (phi_z_w[:, w] + beta) / (n_z + V * beta)
                p_z = p_d_t * p_t_w
                p_z /= np.sum(p_z)
                #new_z = np.random.multinomial(1, p_z).argmax()
                new_z = np.random.choice(len(p_z), 1, p=p_z)[0] 

                # set z as the new topic and increment counts
                z_d_n[d][n] = new_z
                theta_d_z[d][new_z] += 1 #prob / mot
                phi_z_w[new_z, w] += 1 #prob / patient
                n_z[new_z] += 1
 
    norm_theta = theta_d_z.copy()
    ns = np.sum(theta_d_z, axis=1)
    for i in range(ns.shape[0]):
        norm_theta[i, :] /= ns[i]
    #print(np.max(norm_theta))

    return norm_theta
        



def LDA_function_makeblop_form(dataframe_for_LDA,diviser_of_matrix,runrun,number_of_calculation, alpha, beta):
    excluded_patient = 0
    vocabulary  =  list(range(dataframe_for_LDA.shape[1]))
    raw_data_T4 = dataframe_for_LDA.T
    data_T4 = raw_data_T4.T
    t4_ = data_T4.to_numpy()/diviser_of_matrix

    #t4_ = np.around(t4_)
    docs = []
    npatients, nvocabulary = t4_.shape
    for n in range (npatients):
        current_doc = []
        doc = t4_[n,:]
        for i in range(nvocabulary):
            for _ in range(int(doc[i])):
                current_doc.append(i)
        docs.append(current_doc)
                
            

    D = len(docs)        # number of documents
    V = len(vocabulary)  # size of the vocabulary 
    T = runrun            # number of topics

    # the parameter of the Dirichlet prior on the per-document topic distributions  #Faire varier
    # the parameter of the Dirichlet prior on the per-topic word distribution


    z_d_n = [[0 for _ in range(len(d))] for d in docs]  # z_i_j
    theta_d_z = np.zeros((D, T))
    phi_z_w = np.zeros((T, V))
    n_d = np.zeros((D))
    n_z = np.zeros((T))

    ## Initialize the parameters
    # m: doc id
    for d, doc in enumerate(docs):  
        # n: id of word inside document, w: id of the word globally
        for n, w in enumerate(doc):
            # assign a topic randomly to words
            z_d_n[d][n] = int(np.random.randint(T))
            # get the topic for word n in document m
            z = z_d_n[d][n]
            # keep track of our counts
            theta_d_z[d][z] += 1
            phi_z_w[z, w] += 1
            n_z[z] += 1
            n_d[d] += 1

    #for iteration in tqdm(range(number_of_calculation)):
    for iteration in range(number_of_calculation):
        for d, doc in enumerate(docs):
            for n, w in enumerate(doc):
                # get the topic for word n in document m
                z = z_d_n[d][n]

                # decrement counts for word w with associated topic z
                theta_d_z[d][z] -= 1
                phi_z_w[z, w] -= 1
                n_z[z] -= 1

                # sample new topic from a multinomial according to our formular
                p_d_t = (theta_d_z[d] + alpha) / (n_d[d] - 1 + T * alpha)
                p_t_w = (phi_z_w[:, w] + beta) / (n_z + V * beta)
                p_z = p_d_t * p_t_w
                p_z /= np.sum(p_z)
                #new_z = np.random.multinomial(1, p_z).argmax()
                new_z = np.random.choice(len(p_z), 1, p=p_z)[0] 

                # set z as the new topic and increment counts
                z_d_n[d][n] = new_z
                theta_d_z[d][new_z] += 1 #prob / mot
                phi_z_w[new_z, w] += 1 #prob / patient
                n_z[new_z] += 1
 
    norm_theta = theta_d_z.copy()
    ns = np.sum(theta_d_z, axis=1)
    for i in range(ns.shape[0]):
        norm_theta[i, :] /= ns[i]
    #print(np.max(norm_theta))

    return np.round(norm_theta)


def relabeling(number_of_patient_phenotypes,frame_binary_with_threshold ):
    dictionary = {}
    new_label = 0
    for phenotypes in range(0,number_of_patient_phenotypes):
        list_clusters = frame_binary_with_threshold[frame_binary_with_threshold['Phenotype'] == phenotypes]['Hierarchical_clustering'].to_list()
        if phenotypes == 0:
            cluster_number = max(list_clusters, key = list_clusters.count)
            dictionary[cluster_number] = new_label
            new_label =+ 1
        else:
            new_list_of_clusters = list(filter((cluster_number).__ne__, list_clusters))
            dictionary[max(new_list_of_clusters, key = new_list_of_clusters.count)] = new_label
            new_label =+ 1

    return dictionary


def relabeling_hedi_style_V2(real_labels, predicted_label):
    dictionary = {}
    visited_clusters = []
    number_of_clusters = len(np.unique(real_labels))
    for cluster in range(1,number_of_clusters+1):
        idx = np.where(real_labels == cluster)[0]
        lab , count = np.unique(predicted_label[idx], return_counts = True)
        #print(lab,count)
        for c in visited_clusters:
            #ix = lab.index(c)
            ix = np.where(lab == c)[0]
            count = np.delete(count, ix)
            lab = np.delete(lab, ix)
        if len(count) == 0:
            #print('Warning')
            cl = cluster
            visited_clusters.append(cl)
            dictionary[cluster] = cl

        else:
            cl = np.argmax(count)
            visited_clusters.append(lab[cl])
            dictionary[cluster] = lab[cl]
    
    return dictionary


def labeling_datas_Kmeans_KNN(path_to_store_patient_frame,number_of_cluster,n_sample ,path_to_store_frame, strating_point = 0, continuous_data = True):
    if continuous_data == True:
        a = 'Generated_file'
    else:
        a = 'Generated_binary'


    dirs = [path_to_store_patient_frame]
    for dir in dirs:
        #files = glob(f'{dir}/Generated_file_*.csv')
        files = glob(f'{dir}/{a}_*.csv')

        for file in files:
            data = pd.read_csv(file)
            name = '_'.join(file.split("_")[-2:]).replace('.csv','')
            print(file)

            if len(pd.read_csv(file)) > n_sample:
                if strating_point == 0 :
                    row_data = pd.read_csv(file, index_col=0)
                                
                    #data =  np.log10(np.maximum(row_data.sample(n=n_sample, random_state=42),max_val))
                    data = row_data.sample(n=n_sample, random_state=42)
                    subset_data = stats.zscore(data)
                    print(subset_data.shape)
        
                    #Calculate Kmeans Clustering data previsouly reduced with fixed number of cluster
                    time_start = time.time()
                    kmeans = KMeans(n_clusters=number_of_cluster, random_state=41).fit(subset_data)
                    print('Kmeans done: Time elapsed: {} seconds'.format(time.time()-time_start))
                    labels = kmeans.labels_
                    centroids_ref  = kmeans.cluster_centers_

                    counting_occurence_in_cluster_ref = Counter(labels)
                    reference_dataframe = pd.DataFrame.from_dict(counting_occurence_in_cluster_ref, orient='index').reset_index()
                    reference_dataframe = reference_dataframe.rename(columns={'index':'Cluster', 0:f'Count_{name}'})

                    vals_reference = np.fromiter(counting_occurence_in_cluster_ref.values(), dtype=float)

                    strating_point += 1
                        
                else: 
                    
                    row_data_to_compare = pd.read_csv(file, index_col=0)
                    data_to_compare = row_data_to_compare.sample(n=n_sample, random_state=42)
                    
                    subset_data_unref = stats.zscore(data_to_compare)
                    print(subset_data_unref.shape)

                    #Calculate Kmeans Clustering data previsouly reduced with fixed number of cluster
                    time_start = time.time()
                    kmeans_unref = KMeans(n_clusters=number_of_cluster, random_state=41).fit(subset_data_unref)
                    print('Kmeans done: Time elapsed: {} seconds'.format(time.time()-time_start))
                    labels_unref = kmeans_unref.labels_
                    centroids_unref = kmeans_unref.cluster_centers_

                    counting_occurence_in_patient_compare = Counter(labels_unref) 

                    vals_unref = np.fromiter(counting_occurence_in_patient_compare.values(), dtype=float)
                    
                    #COMPARING USING KDTREE
                    k = KDTree(centroids_unref)
                    (dists, idxs) = k.query(centroids_ref)

                    vals_unref[idxs] 

                    reference_dataframe[f'Count_{name}'] = vals_unref[idxs] 

                    print(reference_dataframe.shape, reference_dataframe.columns)

    reference_dataframe.sort_values(by=['Cluster'], inplace=True)
    reference_dataframe.sort_index(axis = 1, ascending = True, inplace=True)

    reference_dataframe.to_csv(path_to_store_frame + f'/Data_for_LDA_from_generate_data_with_n_{number_of_cluster}.csv')
    return reference_dataframe


def continuous_data_files_concat_kmeans_reattached(path_to_store_frame, number_of_cluster,number_of_cell):
    dirs = path_to_store_frame
    a = 'Generated_file'
    files = glob(f'{dirs}/{a}*.csv')
    list_name = []
    list_of_frame_to_append = []
    phenotype_code = []
    for file in tqdm(files):
        name = '_'.join(file.split("_")[-2:]).replace('.csv','')
        list_name.append(name)
        if int(''.join(i for i in name if i.isdigit())) < 25:
            phenotype_code.append(1)
        else:
            phenotype_code.append(2)
        #print(name)
        df_to_generate = pd.read_csv(file, index_col= 0)
        list_of_frame_to_append.append(df_to_generate)
    overall_dataframe = pd.concat(list_of_frame_to_append, ignore_index=False)
    overall_dataframe = pd.DataFrame(stats.zscore(overall_dataframe, nan_policy='omit'))


    kmeans = KMeans(n_clusters=number_of_cluster, random_state=41).fit(overall_dataframe)
    labels = kmeans.labels_
    overall_dataframe['K-means labels'] = labels 

    list_of_dataframe_sliced = []

    start_i = 0
    for i in range(number_of_cell,len(overall_dataframe)+number_of_cell,number_of_cell):
        sliced_frame = overall_dataframe.iloc[start_i:i,:]
        list_of_dataframe_sliced.append(sliced_frame)
        start_i =+ i

    list_of_series = []

    for i,frame in enumerate(list_of_dataframe_sliced):
        counter = Counter(frame['K-means labels'])
        list_of_series.append(pd.Series(counter))

    dataframe_for_LDA = pd.DataFrame(list_of_series, index = list_name )
    dataframe_for_LDA.fillna(0, inplace= True)

    return overall_dataframe, dataframe_for_LDA,phenotype_code


def continuous_data_files_concat_kmeans_reattached_V2(path_to_store_frame, number_of_cluster,number_of_cell):

    dirs = path_to_store_frame
    a = 'Generated_file'
    files = glob(f'{dirs}/{a}*.csv')
    list_name = []
    list_of_frame_to_append = []
    for file in tqdm(files):
        name = '_'.join(file.split("_")[-2:]).replace('.csv','')
        list_name.append(name)
        df_to_generate = pd.read_csv(file, index_col= 0)
        list_of_frame_to_append.append(df_to_generate)
    overall_dataframe = pd.concat(list_of_frame_to_append, ignore_index=False)
    overall_dataframe = pd.DataFrame(stats.zscore(overall_dataframe, nan_policy= 'omit'))

    kmeans = KMeans(n_clusters=number_of_cluster, random_state=41).fit(overall_dataframe)
    labels = kmeans.labels_
    overall_dataframe['K-means labels'] = labels 

    list_of_dataframe_sliced = []

    start_i = 0
    for i in range(number_of_cell,len(overall_dataframe)+number_of_cell,number_of_cell):
        sliced_frame = overall_dataframe.iloc[start_i:i,:]
        list_of_dataframe_sliced.append(sliced_frame)
        start_i =+ i

    list_of_series = []

    for i,frame in enumerate(list_of_dataframe_sliced):
        counter = Counter(frame['K-means labels'])
        list_of_series.append(pd.Series(counter))

    dataframe_for_LDA = pd.DataFrame(list_of_series, index = list_name )
    dataframe_for_LDA.fillna(0, inplace= True)

    return overall_dataframe, dataframe_for_LDA