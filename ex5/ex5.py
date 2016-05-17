from numpy import loadtxt, insert, argsort, sqrt, searchsorted, mean, matrix, array, asarray, ones, where, square, delete, zeros
import matplotlib.pyplot as plot
from scipy import linalg
import random
# Bioinfotools exercise 5

# Put your code instead of the 'pass' statements

	
GE = loadtxt("ge.txt")
GENE_NAMES = loadtxt("gene_names.txt", dtype = str)
CHR1_GENE_NAMES = loadtxt("chr1_genes.txt", dtype = str)
BMI = loadtxt("bmi.txt")
GENDER = loadtxt("gender.txt")
AGE = loadtxt("age.txt")
LIVER_DISEASE = loadtxt("disease_status.txt")
ALL_P =  [1,3,5,10,20,30, 40, 50]
TRAIN_PERCENTAGE = 0.7
K = 15

def average(lst):
	"""
	lst - a list on numbers
	return the average of lst
	"""
	return sum(lst)/len(lst)
	
def standard_deviation(lst):
	"""
	lst - a list on numbers
	retrun the standard deviation of lst
	"""
	import math
	avg = average(lst)
	newlst = [ math.pow(avg - x, 2) for x in lst ]
	return math.sqrt(sum(newlst)/len(lst))
	
def covariance(u, v):
	m_u = u-mean(u)
	m_v = v-mean(v)
	return mean(m_u*m_v)

	
def lin_cor(u, v):
	cov = covariance(u,v)
	sd_u = standard_deviation(u)
	sd_v = standard_deviation(v)
	return cov / (sd_u * sd_v)
	

# def lin_cor(u, v):
#     #substructing the avgs from the vectors
#     u_minus_avg = u - (sum(u)/len(u))
#     v_minus_avg = v - (sum(v)/len(v))
#     #calculating the correlation
#     numerator = sum(u_minus_avg*v_minus_avg)#**2 #calculating the numerator
#     denuminator = (sum(u_minus_avg**2)*sum(v_minus_avg**2))**0.5 #calculating the denuminator
#     #making sure the denuminator isn't zero
#     if denuminator:
#         return numerator/denuminator
#     return 0
    


def squared_lin_cor(u, v):
	return square(lin_cor(u,v))

def knn(X, y, u, K):
	"""
	K nearest neighbors algorithm

	input - 
	X - matrix of n	X P features.
	y - an n X 1 binary labels vector (0 or	1 lable for each sample in X)
  	u - a 1 X P new observation vector
  	K - an integer represents the number of neighbors to consider
	"""
	# find euclidean distance from each sample in X to sample u
	samples_dist = [euclidean_distance(sample, u) for sample in X]
	# find K indices of the closest sample 
	Nk = argsort(samples_dist)[:K]
	return round(sum(y[Nk])/K)



def euclidean_distance(A, B):
    """
    calculates euclidean distance between two n X m matrixes A and B
    Note: Both A and B dimensions are n X m 
    """
    return sqrt(((A - B)**2).sum(axis=0))

	
def remove_duplicate_rows_from_sorted_matrix(data_matrix):
	"""
	removes duplicate entries in UCSC location data (data_matrix)
	returns data_matrix without duplicates
	"""
	keep_indices = set()
	for i in range(data_matrix.shape[0]):
		keep_indices.add(where(data_matrix == data_matrix[i,1])[0][-1])
	
	keep_indices = list(keep_indices)
	return data_matrix[keep_indices]

# for ques a.
def correlation_matrix(data, indices_from_data):
	"""
	returns matrix where S[i,j] is the lin_cor of indicis indices_from_data[i], indices_from_data[j] in data
	"""
	S = matrix([[1.0]*len(indices_from_data)]*len(indices_from_data))
	for i in range(indices_from_data.size):
		for j in range(i, indices_from_data.size):
			S[i,j] = lin_cor(data[:, indices_from_data[i]], data[:, indices_from_data[j]])
			S[j,i] = S[i,j]
	return S

#ques b:

def train_test_choice(X, Y, train_percentage):
	"""
	randomly chooses train_percentage of samples in X to be the X_train set and (1-train_percentage) to be the X_test set
	the sample samples are chosen to Y_train and Y_test
	returns X_train, Xtest, Y_train, Y_test
	"""
	n_samples, m_genes = X.shape 
	# choose train_percentage of samples randomly from n_samples indices as train sampels
	samples_indices = range(n_samples)
	train_smaples_indices = random.sample(samples_indices, round(train_percentage*n_samples))
	test_samples_indices =  delete(samples_indices, train_smaples_indices)
	# extract data of train samples
	X_train = X[train_smaples_indices,:]
	Y_train = Y[train_smaples_indices]
	X_test = X[test_samples_indices,:]
	Y_test = Y[test_samples_indices]
	return X_train, X_test, Y_train, Y_test

def correlation_argsort(u, v):
	"""
	this function calculates squared correlation between every column in u and v
	returns an array containing indices of u columns where the first index is the most correleted column with v
	
	input - 
	u - n X m matrix
	v - n X 1 vector

	"""
	gene_cor = [squared_lin_cor(u[:,j], v) for j in range(u.shape[1])] 
	return  argsort(gene_cor)[::-1]


#c.
def regress(sample_genes, sample_bmi_value):
	"""
	calculates linear regression and returns predictors
	"""
	A = matrix(insert(sample_genes,0,1))
	return linalg.lstsq(A, matrix(sample_bmi_value))[0][:,0]

def train_regression(ge, bmi):
	"""
	this function calculates predicted bmi
	according to the regression model describes in section c/.
	returns predictors for every sample
	"""
	n_samples, m_genes = ge.shape
	predictors = []
	# run regression on every sample
	for sample in range(n_samples):
		predictors.append(regress(ge[sample,:], bmi[sample]))

	# return predictors
	return predictors

def fit_regression(X_test, predictors):
	"""
	tests regression on data
	returns predicted value
	"""
	n_samples, m_genes = X_test.shape
	predicted = []
	for j,sample in enumerate(X_test):
		predicted.append(predictors[j][0] + sum([predictors[j][i+1]*X_test[j,i] for i in range(m_genes)]))
	return predicted

def scatter_plot(x, y, xtitle, ytitle, title, outfile):
    plot.scatter(x, y)
    plot.xlabel(xtitle)
    plot.ylabel(ytitle)
    plot.title(title)
    plot.savefig(outfile)
    plot.close()

def sp_score(observed, predicted):
	"""
	calculates sp score according to the model describes in section c.
	"""
	return squared_lin_cor(observed, predicted)

def evaluate_the_predictability(ge, bmi, all_p, train_percentage):
	"""
	this function returns the evaluation scores of the prediction of bmi using ge.
	otuput plot will be saved to fig2.png

	input - 
	ge - an n samples by m genes GE data matrix
	bmi - observed bmi data (vector of length n)
	train_percentage - percentage of samples to include in training set
	all_p - array containing p values which are the number of best p genes to check the regression model on

	"""
	n_samples, m_genes = ge.shape
	# choose train and test sets 
	train_ge, ge_test, train_bmi, bmi_test = train_test_choice(ge, bmi, train_percentage)
	# sort genes by corellation to bmi
	gene_cor_sorted_indices = correlation_argsort(train_ge, train_bmi)
	
	sp_scores = []
	for p in all_p:
		# find p most correlated genes
		best_p = gene_cor_sorted_indices[:p]
		# train model on training set
		bs = train_regression(train_ge[:,best_p], train_bmi)
		bmi_test_prediction = fit_regression(ge_test[:,best_p], bs)
		sp = sp_score(bmi_test, array(bmi_test_prediction))
		sp_scores.append(sp)

	return sp_scores


#ques f.

def check_accuracy_knn(observations, predictions):
	"""
	returns the accuracy of the knn algorithm
	accuracy is the precemt of correct predictions in the test set ot of total number of test samples
	"""
	correct_counter = 0 
	tests_amount = len(observations)
	for i in range(tests_amount):
		if observations[i] == predictions[i]:
			correct_counter += 1
	return correct_counter/tests_amount


def heatmap(mat, filename):
	"""
	plots a heatmap of matrix mat into filename file
	"""
	plot.imshow(mat)
	plot.colorbar()
	plot.savefig(filename)
	plot.close()

def a():
	"""
	answer question a
	"""
	# ques a.
	chr1_gene_names_loc = loadtxt("UCSC_retrieved_info.txt", dtype = str)
	# get the indices of the genes in chromosome1 in the ge matrix

	# sort gene_names and get the indices of the sorted array
	gene_names_sorted_indices = argsort(GENE_NAMES) 		
	# gene_names sorted	
	gene_names_sorted = GENE_NAMES[gene_names_sorted_indices] 
	# get the indices of genes in chromosome1
	chr1_gene_names_indices = gene_names_sorted_indices[searchsorted(gene_names_sorted, CHR1_GENE_NAMES)] 
	# matrix correlation
	S = correlation_matrix(GE, chr1_gene_names_indices)
	heatmap(S, "chr1")

	# location:
	# remove duplicate genes
	chr1_gene_names_loc = remove_duplicate_rows_from_sorted_matrix(chr1_gene_names_loc)
	# sort by location in chromosome 1
	chr1_gene_names_loc = asarray(sorted(chr1_gene_names_loc, key = lambda x: int(x[0][2:-1])))
	# get indexes from ge matrix to use
	chr1_gene_names_loc_sorted_indices = gene_names_sorted_indices[searchsorted(gene_names_sorted, chr1_gene_names_loc[:,1])]

	S2 = correlation_matrix(GE, chr1_gene_names_loc_sorted_indices)
	heatmap(S2, "chr1_loc")



def c():
	"""
	answer question c
	"""
	n_samples, m_genes = GE.shape
	# predict and evaluate over 20 tries
	best_p_sp_scores =[]
	for i in range(20):
		sp_scores = evaluate_the_predictability(GE, BMI, ALL_P, TRAIN_PERCENTAGE)
		max_sp_score = max(sp_scores)
		max_sp_score_index = sp_scores.index(max_sp_score)
		best_p_sp_scores.append(ALL_P[max_sp_score_index])
	# plot
	scatter_plot(ALL_P, sp_scores, 'P', 'Sp', 'Sp values as a function of P', 'fig2.png')
	
	best_p_avg = average(best_p_sp_scores)
	plot.plot(range(20), best_p_sp_scores)
	plot.plot(range(20),[best_p_avg]*20 , 'r-') 

	plot.xlabel("try number")
	plot.ylabel("best p")
	plot.title("best p for run - average is P = %s"%best_p_avg)
	plot.savefig("OUT")

def f():
	"""
	answer question f
	run knn and clac accuracy
	"""
	# split to train and test sets
	ge_train, ge_test, liver_train, liver_test = train_test_choice(GE, LIVER_DISEASE, TRAIN_PERCENTAGE)
	# run knn
	prediction_lables = [knn(ge_train, liver_train, sample, K) for sample in ge_test] 
	# clac accuracy
	accuracy = check_accuracy_knn(liver_test, prediction_lables)
	return accuracy


def g():
	"""
	answer question g
	impoved knn
	"""
	# use only genes that are correlated with liver disease
	gene_cor_sorted_indices = correlation_argsort(GE, LIVER_DISEASE)
	genes = gene_cor_sorted_indices[:400]
	ge_improved = GE[:,genes]

	# split to train and test sets
	ge_train, ge_test, liver_train, liver_test = train_test_choice(ge_improved, LIVER_DISEASE, TRAIN_PERCENTAGE)
	# run knn
	prediction_lables = [knn(ge_train, liver_train, sample, K) for sample in ge_test] 
	# clac accuracy
	accuracy = check_accuracy_knn(liver_test, prediction_lables)
	return accuracy

def h():
	"""
	answer question f
	return the name of the most correleted to liver disease gene
	"""
	return  GENE_NAMES[correlation_argsort(GE, LIVER_DISEASE)[0]]	
# print(c())
knn_acc = []
[knn_acc.append(f()) for i in range(20)]
knn_acc_avg = average(knn_acc)
plot.plot(range(20), knn_acc)
plot.plot(range(20),[knn_acc_avg]*20 , 'r-') 

plot.xlabel("try number")
plot.ylabel("accuracy")
plot.title("accuracy for run  - average is P = %s"%knn_acc_avg)
plot.savefig("knn")
plot.close()
knn_acc=[]
[knn_acc.append(g()) for i in range(20)]
knn_acc_avg = average(knn_acc)
plot.plot(range(20), knn_acc)
plot.plot(range(20),[knn_acc_avg]*20 , 'r-') 

plot.xlabel("try number")
plot.ylabel("accuracy")
plot.title("IMPROVED - accuracy for run  - average is P = %s"%knn_acc_avg)
plot.savefig("imp_knn")
# print(g())
# print(h())
# print(h())
# print(c())