from bayes_utils import *
import math
import random
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class BinaryBayesModel(object):
    # Basic Bayes Model fro reference
    summaries = 0
    global_summary = 0
    total_len = 0
    class_instances_len = {}
    classSize  = {}
    sens_attr_prop = {}
    labels = []
    disc_attr = []
    theta = 0.9
    class_weights_disc_attr_dict = {}
    class_weights_dict = {}
    decay_factor = 0
    
    
    def train(self, train_data, sens_attr_index):
    
        train_data = uniform_data(train_data)
        self.summaries = discrete_summarize_by_class(train_data, self.class_instances_len)
        
        for i in range(len(train_data)):
        	self.labels.append(train_data[i][-1])
        class_weights = compute_class_weight('balanced', np.unique(self.labels), self.labels)
        
        unique_labels = np.unique(self.labels)
        for n in range(len(unique_labels)):
        	up_dict = {unique_labels[n]:class_weights[n]}			
        	self.class_weights_dict.update(up_dict)
        self.global_summary = discrete_summarize_total(train_data)
        self.total_len = len(train_data)
        self.update_classSize(train_data)
        self.update_sens_attr_prop(train_data, sens_attr_index)
        
        return (self.summaries, self.global_summary, self.total_len)
        
            		
    def online_train(self, train_data, sens_attr_index):
        '''
        #self.online_update_classSize(train_data)
   
        
        #self.online_update_sens_attr_prop(train_data, sens_attr_index)
        #lammbda = self.calculateLammbda(train_data)
        #disc_lammbda = 1
        #if self.total_len > 50:
        #	disc_lammbda = self.calculate_disc_lammbda(train_data, sens_attr_index)
        #k1 = 0
        #if disc_lammbda != 1: 
        #	k1 = self.poissonn(disc_lammbda)
        #k2 = self.poissonn(lammbda)
        #k = k1 + k2
        #print("Umsampling fraction: %s" % k)
        
        if k>0 and lammbda != 1:
        	self.summaries = online_discrete_summarize_by_class(self.summaries, train_data, self.class_instances_len, k)
        	self.global_summary = online_discrete_summarize_total(train_data, self.global_summary, k)
        	self.total_len +=k
        else:
        	self.summaries = online_discrete_summarize_by_class(self.summaries, train_data, self.class_instances_len, 1)
        	self.global_summary = online_discrete_summarize_total(train_data, self.global_summary, 1)
        	self.total_len +=1
        '''
        self.online_update_classSize(train_data)
        self.online_update_sens_attr_prop(train_data, sens_attr_index)
        self.labels.append(train_data[-1])
        unique_disc_attr =[]
        class_weights = compute_class_weight('balanced', np.unique(self.labels), self.labels)
        
        unique_labels = np.unique(self.labels)
        for n in range(len(unique_labels)):
        	up_dict = {unique_labels[n]:class_weights[n]}			
        	self.class_weights_dict.update(up_dict)
        ocis = self.classSize['>50K'] - self.classSize['<=50K']
        
        if ocis>0:
        	self.decay_factor = self.class_weights_dict[train_data[-1]]/1-ocis
        else:
        	self.decay_factor = self.class_weights_dict[train_data[-1]]/1+ocis
        k1 = 1
        i = 0
        j = 0
        m = 0
        k = 1
        for i in range(len(unique_labels)):
        	if train_data[-1]== unique_labels[i]:
        		break
        min_class_weight = max(class_weights)
        for j in range(len(class_weights)):
        	if min_class_weight==class_weights[j]:
        		break
        min_class = unique_labels[j]	
                
        if train_data[-1]== min_class:
        	self.disc_attr.append(train_data[sens_attr_index])
        	class_weights_disc_attr = compute_class_weight('balanced', np.unique(self.disc_attr), self.disc_attr)
        	unique_disc_attr = np.unique(self.disc_attr)
        	min_class_weight_disc_attr = max(class_weights_disc_attr)
        	for m in range(len(class_weights_disc_attr)):
        		if min_class_weight_disc_attr == class_weights_disc_attr[m]:
        			break
        	min_class_disc_attr = unique_disc_attr[m]
        	if train_data[sens_attr_index]==min_class_disc_attr:
        		k1 = class_weights_disc_attr[m]
        for n in range(len(unique_disc_attr)):
        	up_dict = {unique_disc_attr[n]:class_weights_disc_attr[n]}			
        	self.class_weights_disc_attr_dict.update(up_dict)
        #print(self.class_weights_disc_attr_dict)	
        lammbda = self.calculateLammbda(train_data)
        k2 = self.poissonn(lammbda)
        if k>0 and lammbda != 1:
        	k = k2 * k1
        self.summaries = online_discrete_summarize_by_class(self.summaries, train_data, self.class_instances_len, 1)
        
        self.global_summary = online_discrete_summarize_total(train_data, self.global_summary, 1)
        self.total_len +=1	
        return (self.summaries, self.global_summary, self.total_len)
    
    def update_classSize(self, train_data):
        for i in range(len(train_data)):
        	if train_data[i][-1] not in self.classSize:
        	        up_dict = {train_data[i][-1]:0.5}
        	        self.classSize.update(up_dict)
        for classValue in self.classSize:
        	if classValue == train_data[i][-1]:
        		update = self.theta * self.classSize.get(classValue) + (1-self.theta)
        		self.classSize[classValue]= update
        	else:
        		update = self.theta * self.classSize.get(classValue)
        		self.classSize[classValue] = update
    
           		    
    def online_update_classSize(self, train_data):
        if (train_data[-1] not in self.classSize):
        	up_dict = {train_data[-1]:0.5}
        	self.classSize.update(up_dict)
        for classValue in self.classSize:
        	if classValue == train_data[-1]:
        		update = self.theta * self.classSize.get(classValue) + (1-self.theta)
        		self.classSize[classValue] = update
        	else:
        		update = self.theta * self.classSize.get(classValue)
        		self.classSize[classValue] = update
    def update_sens_attr_prop(self, train_data, sens_attr_index):
    
        for i in range (len(train_data)):
        	if (train_data[i][-1] not in self.sens_attr_prop):
        		self.sens_attr_prop[train_data[i][-1]] = {train_data[i][sens_attr_index]:0.5}
        	else:
                      		label = train_data[i][-1]
                      		gender = train_data[i][sens_attr_index]
                      		if gender not in self.sens_attr_prop[label]:
                      			up_dict = {train_data[i][sens_attr_index]:0.5}
                      			self.sens_attr_prop[label].update(up_dict)
         
        for label in self.sens_attr_prop:
        	for gender in self.sens_attr_prop[label]:
        		if gender == train_data[i][sens_attr_index] and label == train_data[i][-1]:
        			update = self.theta * self.sens_attr_prop[label].get(gender) + (1-self.theta)
        			up_dict = {gender: update}
        			self.sens_attr_prop[label].update(up_dict)
        		else:
        			update = self.theta * self.sens_attr_prop[label].get(gender)
        			up_dict = {gender: update}
        			self.sens_attr_prop[label].update(up_dict)
        
        		
    def online_update_sens_attr_prop(self, train_data, sens_attr_index):
    	
    	if (train_data[-1] not in self.sens_attr_prop):
    		self.sens_attr_prop[train_data[-1]] = {train_data[sens_attr_index]:0.5}
    	else:
                label = train_data[-1]
                gender = train_data[sens_attr_index]
                if gender not in self.sens_attr_prop[label]:
                	up_dict = {train_data[sens_attr_index]:0.5}
                	self.sens_attr_prop[label].update(up_dict)
    	for label in self.sens_attr_prop:
        	for gender in self.sens_attr_prop[label]:
        		if gender == train_data[sens_attr_index] and label == train_data[-1]:
        			update = self.theta * self.sens_attr_prop[label].get(gender) + (1-self.theta)
        			up_dict = {gender: update}
        			self.sens_attr_prop[label].update(up_dict)
        		else:
        			update = self.theta * self.sens_attr_prop[label].get(gender)
        			up_dict = {gender: update}
        			self.sens_attr_prop[label].update(up_dict)
    	
    	
        
    def getMajorityClass(self):
        v = list(self.classSize.values())
        k = list(self.classSize.keys())
        return k[v.index(max(v))]
        
    def getMinorityClass(self):
        v = list(self.classSize.values())
        k = list(self.classSize.keys())
        return k[v.index(min(v))]
           
    def calculateLammbda(self, train_data):
        majClass = self.getMajorityClass()
        lammbda = self.classSize.get(majClass)/self.classSize.get(train_data[-1])
        return lammbda
    
    def calculate_disc_lammbda(self, train_data, sens_attr_index):
        majClass = self.getMajorityClass()
        minClass  = self.getMinorityClass()
        disc_lammbda = 1
        if train_data[-1]== minClass:
        	v = list(self.sens_attr_prop[minClass].values())
        	k = list(self.sens_attr_prop[minClass].keys())
        	maj_gender = k[v.index(max(v))]
        	label = train_data[-1]
        	gender = train_data[sens_attr_index]
        	
        	disc_lammbda = self.sens_attr_prop[label].get(maj_gender)/ self.sens_attr_prop[label].get(gender) 
        	
        return disc_lammbda
    def poissonn(self, lammbda):
        if lammbda<100:
        	product = 1.0
        	sum = 1.0
        	threshold = np.random.uniform(0,1,1) * math.exp(lammbda)
        	maximum = max(100, 10 * math.ceil(lammbda))
        	i = 1
        	while i<maximum and sum < threshold:
        		product *= (lammbda/i)
        		sum+=product
        		i = i+1
        	return i-1
        x = lammbda + math.sqrt(lammbda) * random.gauss(0,1)
        if x<0:
        	return 0;
        return int(math.floor(x))
    
    
    def evaluate(self,
                 input, summaries=None,
                 global_summary=None, total_len=None, class_weights_disc_attr_dict = None):
        # Evaluates the probabilies of input beloging to each set of
        # classes

        if not summaries:
            summaries = self.summaries

        if not global_summary:
            global_summary = self.global_summary

        if not total_len:
            total_len = self.total_len
        if not class_weights_disc_attr_dict:
        	class_weights_disc_attr_dict = self.class_weights_disc_attr_dict
       
        
        		
        '''
        print(decay_factor)
        print(self.class_weights_disc_attr_dict)
        print(input[9])
        '''
        class_weights_dict = self.class_weights_dict
        d = self.decay_factor
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            summary = classSummaries[0]
            total = classSummaries[1]
            probabilities[classValue] = float(1)
	
            missing = '?'
            ocis = self.classSize['>50K'] - self.classSize['<=50K']
            
            if ocis>0:
            	self.decay_factor = class_weights_dict[classValue]/(1-ocis)
            else:
            	self.decay_factor = class_weights_dict[classValue]/(1+ocis)
            
            for i in range(len(summary)):
                x = input[i]
                
                numerator = float(summary[i].get(x, 0)) + 2*(float(global_summary[0][i].get(x, 0))/float(1+global_summary[0][i].get(missing, 0)))
                
                denominator = float(summary[i].get(missing, 0)) + 2
                p_i_class = numerator/denominator
            #    p_i_class = \
             #       float(summary[i].get(x, 0))/float(total) * \
              #      (float(global_summary[0][i].get(x, 0))/float(total_len))
                probabilities[classValue] *= float(p_i_class)
            #probabilities[classValue]*=self.decay_factor
            
            probabilities[classValue] = probabilities[classValue] * ((total+1)/(global_summary[1]+2))
        return probabilities

    def predict(self, input, summary = None, global_summary=None):
    	
        probabilities = self.evaluate(input, summary, global_summary)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        '''
        print(self.class_weights_disc_attr_dict)
        print(input[9])
        print(probabilities)
        print(bestProb)
        print(bestLabel)
        '''
        return bestLabel

    def getPredictions(self, testSet, summary = None, global_summary=None):
        predictions = []
        for i in range(len(testSet)):
            result = self.predict(testSet[i], summary, global_summary)
            predictions.append(result)
        return predictions
    
   

    def test(self, test_data, summary = None, global_summary=None):
        def _getAccuracy(testSet, predictions):
            correct = 0
            maj_tp = 0
            maj_fn = 0
            min_tp = 0
            min_fn = 0
            maj_fp = 0
            min_fp = 0
            maj_tn = 0
            min_tn = 0
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            #>50K is positive class
            for i in range(len(test_data)):
                if test_data[i][-1] == predictions[i]:
                    correct += 1
                if test_data[i][-1] == '>50K' and predictions[i] == '>50K':
                	tp +=1
                elif test_data[i][-1] == '>50K' and predictions[i] == '<=50K':
                	fn +=1
                elif test_data[i][-1] == '<=50K' and predictions[i] == '>50K':
                	fp +=1
                elif test_data[i][-1] == '<=50K' and predictions[i] == '<=50K':
                	tn +=1
                	
       	         
        	            
            recall = tp/(tp+fn) 
            print("tp: %s " % tp)
            print("fp: %s " % fp)
            print("tn: %s " % tn)
            print("fn: %s " % fn)
            print("class_len: %s" % self.class_instances_len)
            print("recall: %s" % recall)
            '''maj_recall = (maj_tp/(maj_tp+min_fn))*100
            print("Recall for male group: %s " % maj_recall)
            
            min_recall = (min_tp/(min_tp+min_fn))*100
            print("Recall for female group: %s " % min_recall)
            
            print("fp for maj group: %s " % maj_fp)
            print("fp for min group: %s " % min_fp)
            
            print("tn for maj group: %s " % maj_tn)
            print("tn for min group: %s " % min_tn)'''
            
            '''total_recall = ((female_tp+male_tp)/(female_tp+male_tp+female_fn+male_fn))*100
            print("Total Recall: %s " % total_recall)'''
            
            accuracy = (correct/float(len(test_data))) * 100.0
            
            return accuracy

        predictions = self.getPredictions(test_data, summary, global_summary)
        accuracy = _getAccuracy(test_data, predictions)
        return (accuracy, predictions)

    def discrimination_measure(self, index, label, test_data, summary=None, global_summary=None):
        
        predictions = self.getPredictions(test_data, summary, global_summary)

        test_global_summary = (discrete_summarize_total(test_data),len(test_data))
        
        results = {}

        possible_values = list(
            set([sample[index] for sample in test_data]))
        for possible_value in possible_values:
            results[possible_value] = 0

        for i in range(0, len(predictions)):
            if predictions[i] == label:
                results[test_data[i][index]] += 1

        
        for key, value in test_global_summary[0][index].items():
            results[key] = float(results[key])/value

        # warning, this only supports 2 classes for now:

        values = results.values()

        return max(values) - min(values)


class SplitFairBayesModel(BinaryBayesModel):
    # This model splits the model into sub models individually, for each
    # value of a sensitive variable.

    sensitive_params_summaries = {}
    sensitve_param_indexes = []
    class_instances_len = {'Male':{'<=50K': 0, '>50K':0}, 'Female': {'<=50K': 0, '>50K':0} }
    def __init__(self, sensitive_parameter_indexes):
        super(SplitFairBayesModel, self).__init__()
        self.sensitve_param_indexes = sensitive_parameter_indexes
	 
    def train(self, train_data):
        # WARNING, ONLY WORKS CURRENTY WITH ONE SENSITIVE PARAMETER.
        # Lets train a model for each sensitive parameters
        for index in self.sensitve_param_indexes:
            self.sensitive_params_summaries[index] = {}
            # Get all unique values for each of the sensitive parameters
            # making a set, and getting back to list does the trick
            possible_values = list(
                set([sample[index] for sample in train_data]))

            for possible_value in possible_values:
                # TODO: Copy the list, pop and buld recursively this thing.
                data_set_partition = list(filter(
                    lambda x: x[index] == possible_value, train_data))
		
               
                for i in range(len(train_data)):
                	self.class_instances_len[possible_value][train_data[i][-1].lstrip()]= self.class_instances_len[possible_value][train_data[i][-1].lstrip()]+1
                self.sensitive_params_summaries[index][possible_value] = \
                    super(SplitFairBayesModel, self).train(data_set_partition)
                    
    def online_train(self, train_data):
        # WARNING, ONLY WORKS CURRENTY WITH ONE SENSITIVE PARAMETER.
        # Lets train a model for each sensitive parameters
        for index in self.sensitve_param_indexes:
           # self.sensitive_params_summaries[index] = {}
            # Get all unique values for each of the sensitive parameters
            # making a set, and getting back to list does the trick
            possible_value = train_data[9]
            self.class_instances_len[possible_value][train_data[-1].lstrip()]= self.class_instances_len[possible_value][train_data[-1].lstrip()]+1
            self.sensitive_params_summaries[index][possible_value] = \
                    super(SplitFairBayesModel, self).online_train(train_data)

    def predict(self, input):
        # Decide what to model to use:
        # WARNING! THIS ALSO ONLY SUPPORTS ONE SENSITIVE VARIABLE !!
        sensitive_index = self.sensitve_param_indexes[0]
        sentivive_value = input[sensitive_index]

        (summaries, global_summary, total_len) = \
            self.sensitive_params_summaries[sensitive_index][sentivive_value]
        probabilities = \
            self.evaluate(input, summaries=summaries,
                          global_summary=global_summary, total_len=total_len)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel


class BalancedBayesModel(BinaryBayesModel):
    # WARNING, this is a specific implementation for the dataset
    # following the paper Calders10

    def discrimination_measure(
            self, index, discriminated_class, privileged_class,
            positive_label, test_data):
        # An asimteric disctimination measure
        predictions = self.getPredictions(test_data)

        test_global_summary = discrete_summarize_total(test_data)
        results = {}
        total = {}

        possible_values = list(
            set([sample[index] for sample in test_data]))
        for possible_value in possible_values:
            results[possible_value] = 0
            total[possible_value] = 0

        for i in range(0, len(predictions)):
            if predictions[i] == positive_label:
                results[test_data[i][index]] += 1
                total[test_data[i][index]] += 1

        for key, value in test_global_summary[0][index].items():
            results[key] = float(results[key])/value

        # warning, this only supports 2 classes for now:
        # return the pair - (Discrimination score , Total positive labels)
        return (
            results[privileged_class] - results[discriminated_class],
            total[privileged_class] + total[discriminated_class])

    def balance_model(
            self, index, discriminated_class, privileged_class,
            positive_label, negative_label, train_data):
        balance_resutls = []

        total_positive_labels = len(list(
            filter(lambda x: x[index] == positive_label, train_data)))

        (disc, assinged_labels) = \
            self.discrimination_measure(index, discriminated_class,
                                        privileged_class, positive_label,
                                        train_data)

        accuracy = self.test(train_data)[0]
        balance_resutls.append((disc, accuracy))

        while disc > 0:
            if assinged_labels > total_positive_labels:
                self.summaries[positive_label][0][index][discriminated_class] = \
                    self.summaries[positive_label][0][index][discriminated_class] + \
                    0.01 * self.summaries[negative_label][0][index][privileged_class]

                self.summaries[negative_label][0][index][privileged_class] = \
                    self.summaries[positive_label][0][index][discriminated_class] - \
                    0.01 * self.summaries[negative_label][0][index][privileged_class]
            else:
                self.summaries[negative_label][0][index][privileged_class] = \
                    self.summaries[negative_label][0][index][privileged_class] + \
                    0.01 * self.summaries[positive_label][0][index][discriminated_class]

                self.summaries[positive_label][0][index][privileged_class] = \
                    self.summaries[negative_label][0][index][privileged_class] - \
                    0.01 * self.summaries[positive_label][0][index][discriminated_class]

            accuracy = self.test(train_data)[0]
            balance_resutls.append((disc, accuracy))

            (disc, assinged_labels) = \
                self.discrimination_measure(index, discriminated_class,
                                            privileged_class, positive_label,
                                            train_data)

        return balance_resutls
