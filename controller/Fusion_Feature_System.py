from collections import defaultdict
import numpy as np
from itertools import product
import operator
import scipy
from scipy import spatial 


class Fusion_Feature:

    def __init__(self,lenght,type_binary, number_biom_enrol, bio1, bio2, bio3 = None):

        self.binning = {}

        self.template_code = {}

        self.map_enrol = {}

        self.normalizer_bio1 = {}
        
        self.normalizer_bio2 = {}
        
        self.normalizer_bio3 = {} 

        self.lenght = lenght

        self.number_bins_enrolment = 0

        self.type_binary = type_binary

        self.number_biom_enrol = number_biom_enrol

        self.bio1 = bio1

        self.bio2 = bio2

        self.bio3 = bio3

        self.dict_normalizer_face_baseline = {'mean': 0.3051281057062831,'std': 0.05574250472628351}

        self.dict_normalizer_fingerprint_baseline = {'mean': 0.18334383632596685,'std': 0.15586952928405295}

        self.dict_normalizer_iris_baseline = {'mean': 0.4552527286902287,'std': 0.10094153035507722}


        self.dict_normalizer_face_biohashing = {'mean': 0.3611796082910886,'std': 0.10547936428704169}

        self.dict_normalizer_fingerprint_biohashing = {'mean': 0.08770416091160221,'std': 0.07475547039618236}

        self.dict_normalizer_iris_biohashing = {'mean': 0.45507678448867533,'std': 0.10277504905290015}


        self.dict_normalizer_face_grp = {'mean': 0.16389176383677848,'std': 0.07207417500985165}

        self.dict_normalizer_fingerprint_grp = {'mean': 0.6716105139916728,'std': 0.20990935616675782}

        self.dict_normalizer_iris_grp = {'mean': 0.06886883008239504 ,'std': 0.09083807203838756}

        self._initializing_normalization_bio()

        self._generate_template_code()


    def _initializing_normalization_bio(self):

        if self.number_biom_enrol == 2:

            if self.type_binary == 'baseline':

                if 'Faces' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_face_baseline
                if 'Iris' in self.bio1:
                        self.normalizer_bio1 = self.dict_normalizer_iris_baseline
                if 'Fingerprint' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_fingerprint_baseline
            
                if 'Faces' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_face_baseline
                if 'Iris' in self.bio2:
                        self.normalizer_bio2 = self.dict_normalizer_iris_baseline
                if 'Fingerprint' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_fingerprint_baseline
                
            
            elif self.type_binary == 'biohashing':

                if 'Faces' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_face_biohashing
                if 'Iris' in self.bio1:
                        self.normalizer_bio1 = self.dict_normalizer_iris_biohashing
                if 'Fingerprint' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_fingerprint_biohashing
            
                if 'Faces' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_face_biohashing
                if 'Iris' in self.bio2:
                        self.normalizer_bio2 = self.dict_normalizer_iris_biohashing
                if 'Fingerprint' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_fingerprint_biohashing


            elif self.type_binary == 'grp':

                if 'Faces' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_face_grp
                if 'Iris' in self.bio1:
                        self.normalizer_bio1 = self.dict_normalizer_iris_grp
                if 'Fingerprint' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_fingerprint_grp
            
                if 'Faces' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_face_grp
                if 'Iris' in self.bio2:
                        self.normalizer_bio2 = self.dict_normalizer_iris_grp
                if 'Fingerprint' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_fingerprint_grp
             
        else:

            if self.type_binary == 'baseline':

                if 'Faces' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_face_baseline
                if 'Iris' in self.bio1:
                        self.normalizer_bio1 = self.dict_normalizer_iris_baseline
                if 'Fingerprint' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_fingerprint_baseline
            
                if 'Faces' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_face_baseline
                if 'Iris' in self.bio2:
                        self.normalizer_bio2 = self.dict_normalizer_iris_baseline
                if 'Fingerprint' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_fingerprint_baseline
                
                if 'Faces' in self.bio3:
                    self.normalizer_bio3 = self.dict_normalizer_face_baseline
                if 'Iris' in self.bio3:
                        self.normalizer_bio3 = self.dict_normalizer_iris_baseline
                if 'Fingerprint' in self.bio3:
                    self.normalizer_bio3 = self.dict_normalizer_fingerprint_baseline
            
            elif self.type_binary == 'biohashing':

                if 'Faces' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_face_biohashing
                if 'Iris' in self.bio1:
                        self.normalizer_bio1 = self.dict_normalizer_iris_biohashing
                if 'Fingerprint' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_fingerprint_biohashing
            
                if 'Faces' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_face_biohashing
                if 'Iris' in self.bio2:
                        self.normalizer_bio2 = self.dict_normalizer_iris_biohashing
                if 'Fingerprint' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_fingerprint_biohashing
                
                if 'Faces' in self.bio3:
                    self.normalizer_bio3 = self.dict_normalizer_face_biohashing
                if 'Iris' in self.bio3:
                        self.normalizer_bio3 = self.dict_normalizer_iris_biohashing
                if 'Fingerprint' in self.bio3:
                    self.normalizer_bio3 = self.dict_normalizer_fingerprint_biohashing

            elif self.type_binary == 'grp':

                if 'Faces' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_face_grp
                if 'Iris' in self.bio1:
                        self.normalizer_bio1 = self.dict_normalizer_iris_grp
                if 'Fingerprint' in self.bio1:
                    self.normalizer_bio1 = self.dict_normalizer_fingerprint_grp
            
                if 'Faces' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_face_grp
                if 'Iris' in self.bio2:
                        self.normalizer_bio2 = self.dict_normalizer_iris_grp
                if 'Fingerprint' in self.bio2:
                    self.normalizer_bio2 = self.dict_normalizer_fingerprint_grp
                
                if 'Faces' in self.bio3:
                    self.normalizer_bio3 = self.dict_normalizer_face_grp
                if 'Iris' in self.bio3:
                        self.normalizer_bio3 = self.dict_normalizer_iris_grp
                if 'Fingerprint' in self.bio3:
                    self.normalizer_bio3 = self.dict_normalizer_fingerprint_grp


    def _generate_template_code(self):

        dict_combinations = {}
  
        for i , c in enumerate(product(range(2), repeat=self.lenght)):

            dict_combinations[c] = i
        
        keys = list(dict_combinations.keys())

        for code in keys:

            pattern = ''

            for bit in code:

                pattern += str(bit).strip()
            
            self.template_code[pattern] = dict_combinations[code]


    def mapping_enrol(self, binning):

        self.map_enrol = defaultdict(list)

        for bin_k in binning:

            key_map = self.template_code[bin_k] 

            self.map_enrol[key_map] = binning[bin_k]

            self.number_bins_enrolment = len(self.map_enrol.keys())
        
    
    def mapping_search(self,list_codes):

        list_map_codes_s = []

        for code in list_codes:

            map = self.template_code[code]

            list_map_codes_s.append(map)

        return list_map_codes_s



    def mean_confidence_interval(self,input_list, confidence=0.95):

        a = 1.0 * np.array(input_list)

        n = len(a)

        m, se = np.mean(a), scipy.stats.sem(a)

        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

        return m, h


    def search(self, code):

        candidates = []

        if code in self.map_enrol:

            candidates = self.map_enrol[code]

        else:

             candidates = []

        return candidates
    

    
    """Compators functions"""


    def hamming_comparison(self, feat_s, list_feat):

        list_scores = []

        list_labels = []

        for feat_e in list_feat:

            value = spatial.distance.hamming(feat_e, feat_s)  

            list_scores.append(value)

            # list_labels.append(feat_e[1])

        return list_scores 
    
        
    def comparison_urp(self, feat_s, list_feat):

        list_scores = []

        list_labels = []

        for feat_e in list_feat:

            compare = (feat_e[0] == feat_s)

            value = compare.sum()/float(len(feat_e[0]))

            list_scores.append(value)

            list_labels.append(feat_e[1])

        
        return list_scores,list_labels

    
    def comparison_grp(self, feat_s, list_feat):

        list_scores = []

        list_labels = []

        for feat_e in list_feat:

            compare = (feat_e == feat_s)

            value = compare.sum()/float(len(feat_e)+len(feat_s)-compare.sum())

            list_scores.append(value)

            # list_labels.append(feat_e[1])
        
        return list_scores
    
    def search_open_set_old(self,list_short_codes,feat_s):

        score = -1

        list_scores_total = []

        comparison_type = ""

        number_comparisons = 0

        for code in list_short_codes:

            list_scores = []

            if code in self.map_enrol:

                list_feat = self.map_enrol[code]

                number_comparisons += len(list_feat)
            
                # compare feat against list of enrolment by using the comparator according to the type of binary 
                if self.type_binary == 'baseline' or self.type_binary == 'biohashing' or self.type_binary == 'mlp':

                    comparison_type = "hamming"

                    #comparison is with hamming distance
                    list_scores = self.hamming_comparison(feat_s, list_feat)

                    [list_scores_total.append(s) for s in list_scores]

                    # list_scores = sorted(list_scores, reverse=False)

                    # score = np.round(list_scores[0], 6) 
                
                elif self.type_binary == 'urp':

                    comparison_type = "urp"

                    #comparison is with similarity
                    list_scores = self.comparison_urp(feat_s, list_feat)

                    [list_scores_total.append(s) for s in list_scores]

                    # list_scores = sorted(list_scores, reverse=True)

                    # score = np.round(list_scores[0], 6)                   
                    
                elif self.type_binary == 'grp':

                    comparison_type = "grp"

                    #comparison is with similarity
                    list_scores = self.comparison_grp(feat_s, list_feat) 

                    [list_scores_total.append(s) for s in list_scores]

                    # list_scores = sorted(list_scores, reverse=True)

                    # score = np.round(list_scores[0], 6) 
        
        if comparison_type == 'hamming':

            list_scores_total = sorted(list_scores_total, reverse=False)

            score = np.round(list_scores_total[0], 6) 
        
        elif comparison_type == "urp" or comparison_type == "grp":

            list_scores_total = sorted(list_scores_total, reverse=True)

            score = np.round(list_scores_total[0], 6) 

        return score, number_comparisons



    def generic_compare(self,probe,candidates_list):

        list_scores = []

        list_labels_e = []

        if self.type_binary == 'baseline' or self.type_binary == 'biohashing' or self.type_binary == 'mlp':

            list_scores,list_labels_e = self.hamming_comparison(probe, candidates_list)

            list_total = list(zip(list_scores,list_labels_e))

            list_total.sort(key = lambda i:i[0], reverse = False)

            list_scores,list_labels_e = zip(*list_total)

        elif self.type_binary == 'grp':

            list_scores,list_labels_e = self.comparison_grp(probe, candidates_list) 

            list_total = list(zip(list_scores,list_labels_e))

            list_total.sort(key = lambda i:i[0], reverse = True)

            list_scores,list_labels_e = zip(*list_total)
        
        elif self.type_binary == 'urp':

            list_scores,list_labels_e = self.comparison_urp(probe, candidates_list) 

            list_total = list(zip(list_scores,list_labels_e))

            list_total.sort(key = lambda i:i[0], reverse = True)

            list_scores,list_labels_e = zip(*list_total)

        return list_scores, list_labels_e

    def search_code_cand(self, code,feat_value): # search in enrolment for a single code from probe 

        candidates_lables = []

        candidates_scores = []

        if code in self.map_enrol:

            candidates = self.map_enrol[code]

            candidates_scores,candidates_lables = self.generic_compare(feat_value,candidates)
        
        else:

            candidates_lables = []

            candidates_scores = []


        return candidates_lables,candidates_scores

    
    def concatenation_features_subjects(self,bio1,bio2,bio3,number_bio):

        # feat_bio1 = np.load(bio1)

        # feat_bio2 = np.load(bio2)

        feat_bio1 = bio1

        feat_bio2 = bio2

        if number_bio == 3:

            # feat_bio3 = np.load(bio3)

            concatenated_feat = np.concatenate((feat_bio1, feat_bio2, bio3), axis=None)

            return concatenated_feat
        
        else:

            concatenated_feat = np.concatenate((feat_bio1, feat_bio2), axis=None)

            return concatenated_feat
    

    def __enrol_multi_binning_concat_feat(self,code,value):

        key_map = self.template_code[code] 

        if code in self.binning:

            self.binning[code].append(value)

            self.map_enrol[key_map].append(value)

        else:

            self.binning[code] = [value] 

            self.map_enrol[key_map] = [value]
        
        # self.mapping_enrol(self.binning)


    
    def save_binning_concat_feat(self, max_code, label1,label2,label3,number_bio):

        res = ()

        if number_bio == 1:

            res = (label1)

        elif number_bio == 2:

            res = (label1,label2)
        
        elif number_bio == 3:

            res = (label1,label2,label3)
        
        self.__enrol_multi_binning_concat_feat(max_code,res)
    
    def save_binning_concat_feat_open_set(self, max_code, bio1,bio2,bio3,number_bio):

        res = ()

        if number_bio == 1:

            res = (bio1)

        elif number_bio == 2:

            res = (bio1,bio2)
        
        elif number_bio == 3:

            res = (bio1,bio2,bio3)
        
        self.__enrol_multi_binning_concat_feat(max_code,res)

    
    def normalization_z_score(self, item, normalizer):

        return (item - normalizer['mean'])/normalizer['std']

    
    def compare_normalizer(self, list_feat, feat_s,normalizer_bio):

        list_scores_total = []

        if self.type_binary == 'baseline' or self.type_binary == 'biohashing' or self.type_binary == 'mlp':

            comparison_type = "hamming"

            #comparison is with hamming distance
            list_scores = self.hamming_comparison(feat_s, list_feat)

            normalised_scores = self.normalization_z_score(list_scores,normalizer_bio)

            list_scores_total = normalised_scores

            list_scores_total.sort()
        
        elif self.type_binary == 'grp':

            comparison_type = "grp"

            #comparison is with similarity
            list_scores = self.comparison_grp(feat_s, list_feat) 

            normalised_scores = self.normalization_z_score(list_scores,normalizer_bio)

            list_scores_total = normalised_scores

            list_scores_total.sort(reverse=True)

        return list_scores_total


    def compare(self, list_feat, feat_s):

        list_scores_total = []

        if self.type_binary == 'baseline' or self.type_binary == 'biohashing' or self.type_binary == 'mlp':

            comparison_type = "hamming"

            #comparison is with hamming distance
            list_scores = self.hamming_comparison(feat_s, list_feat)

            list_scores_total = list_scores

            list_scores_total.sort()
        
        elif self.type_binary == 'grp':

            comparison_type = "grp"

            #comparison is with similarity
            list_scores = self.comparison_grp(feat_s, list_feat) 

            list_scores_total = list_scores

            list_scores_total.sort(reverse=True)

        return list_scores_total



    


















            