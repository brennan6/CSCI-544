import numpy as np
import pandas as pd
import json
import warnings
import sys
from platform import python_version

"""Python Script to run HW3 - CSCI544"""

### Python Run Version: 3.9.5
### READ: Include path to dataset as imput to .py file

class run_hw(object):
    """Create this object and run components to print results as requested"""

    def __init__(self, data_fp):
        self.data_fp = data_fp
        self.train_df = None
        self.dev_df = None
        self.test_df = None

        self.cnt_d = None
        self.cnt_d_sorted = None
        self.unknown_word_lst = None
        self.threshold = 2

        self.transition_d = None 
        self.transition_d_formatted = None
        self.emission_d = None 
        self.emission_d_formatted = None
        self.pos_total_counts_d = None

        self.most_likely_start_prob_d = None
        self.greedy_dev_pred_lst = None
        self.greedy_test_pred_lst = None
        self.viterbi_dev_pred_lst = None
        self.viterbi_test_pred_lst = None 
        
    def read_data(self):
        """Method call to read in the data given the folder"""
        tr_headers = ["index", "word", "pos_tag"]
        self.train_df = pd.read_csv(self.data_fp + "/train", sep="\t", header=None)
        self.train_df.columns = tr_headers

        self.dev_df = pd.read_csv(self.data_fp + "/dev", sep="\t", header=None)
        self.dev_df.columns = tr_headers 

        test_headers = ["index", "word"]
        self.test_df = pd.read_csv(self.data_fp + "/test", sep="\t", header=None)
        self.test_df.columns = test_headers

    def clean_data(self):
        """Cleans data by assigning numeric values to <num> word type"""
        # Slight cleaning on num:
        self.train_df["word"] = self.train_df["word"].str.replace(r'^\d+|.\d+$', "<num>", regex=True)
        self.dev_df["word"] = self.dev_df["word"].str.replace(r'^\d+|.\d+$', "<num>", regex=True)
        self.test_df["word"] = self.test_df["word"].str.replace(r'^\d+|.\d+$', "<num>", regex=True)

    def get_word_count(self):
        # Get the count of each word:
        # word-type = word
        self.cnt_d = {}
        for row in self.train_df.iterrows():
            if row[1]["word"] in self.cnt_d:
                self.cnt_d[row[1]["word"]] += 1
            else:
                self.cnt_d[row[1]["word"]] = 1

    def create_unknown_key(self):
        # Create unknown key:
        #No threshold = 1
        unknown_cnt = 0
        self.unknown_word_lst = []   #We want to keep track of unknown words but group together
        for k, v in self.cnt_d.items():
            if v < self.threshold:
                unknown_cnt += v
                self.unknown_word_lst.append(k)
            else:
                continue
        self.cnt_d["< unk >"] = unknown_cnt

    def sort_count_d(self):
        # Sort the occurences in descending order:
        self.cnt_d_sorted = {k: v for k, v in sorted(self.cnt_d.items(), key=lambda item: -item[1])}

    def write_vocab(self):
        ### Write the vocab to vocab.txt
        #punctuation and numbers also count as being part of vocabulary
        i = 0
        with open('vocab.txt', 'w') as f:
            f.write("< unk >")
            f.write("\t")
            f.write(str(i))
            f.write("\t")
            f.write(str(self.cnt_d_sorted["< unk >"]))
            f.write("\n")
            i+=1
            for k, v in self.cnt_d_sorted.items():
                if k == "< unk >":
                    continue
                elif v >= self.threshold:
                    f.write(k)
                    f.write("\t")
                    f.write(str(i))
                    f.write("\t")
                    f.write(str(v))
                    f.write("\n")
                    i+=1
    
    def print_results_1(self):
        """Print the results that need to be outputted for task 1"""
        print("Task 1 Results:")
        print("Selected threshold:", self.threshold)
        print("Total Size of Vocab:", len(self.cnt_d_sorted) - len(self.unknown_word_lst))
        print("Total occurences of < unk >:", self.cnt_d_sorted["< unk >"])
        print("\n")

    def create_transition_prob(self):
        ### Transition Prob. must create prob of all pos_tag transitions, contain in dictionary:
        pos_tag_d = {}
        all_pos_tags = self.train_df["pos_tag"].values
        for i, pos_tag in enumerate(all_pos_tags):
            if i == (len(all_pos_tags) - 1):
                break
            
            next_tag = all_pos_tags[i+1]
            if pos_tag in pos_tag_d:
                if next_tag in pos_tag_d[pos_tag]:
                    pos_tag_d[pos_tag][next_tag] += 1
                else:
                    pos_tag_d[pos_tag][next_tag] = 1
                    
            else:
                pos_tag_d[pos_tag] = {}
                pos_tag_d[pos_tag][next_tag] = 1

        # Need count of transition state individually:
        self.pos_total_counts_d = {}
        for k, v in pos_tag_d.items():
            self.pos_total_counts_d[k] = 0
            for k_inner, v_inner in v.items():
                self.pos_total_counts_d[k] += v_inner
        
        # Create transtion prob:
        self.transition_d = {}
        for k, v in pos_tag_d.items():
            self.transition_d[k] = {}
            for k_inner, v_inner in v.items():
                self.transition_d[k][k_inner] = v_inner/self.pos_total_counts_d[k]
        
        # Format transition d as wanted:
        self.transition_d_formatted = {}
        for k, v in self.transition_d.items():
            for k_inner, v_inner in v.items():
                key_str = str(k) + ", " + str(k_inner)
                self.transition_d_formatted[key_str] = self.transition_d[k][k_inner]

    def create_emission_prob(self):
        ### Emission Prob - Must create prob of word given POS tag. Check if word is in Unknown Lst:
        #Takes ~3 min to run
        pos_to_word_d = {}
        all_pos_tags = self.train_df["pos_tag"].values
        all_words = self.train_df["word"].values
        for i, pos_tag in enumerate(all_pos_tags):
            word = all_words[i]
            if pos_tag in pos_to_word_d:
                if word in self.unknown_word_lst:
                    if "< unk >" in pos_to_word_d[pos_tag]:
                        pos_to_word_d[pos_tag]["< unk >"] += 1
                    else:
                        pos_to_word_d[pos_tag]["< unk >"] = 1
                else:
                    if word in pos_to_word_d[pos_tag]:
                        pos_to_word_d[pos_tag][word] += 1
                    else:
                        pos_to_word_d[pos_tag][word] = 1
                    
            else:
                pos_to_word_d[pos_tag] = {}
                pos_to_word_d[pos_tag][word] = 1
        
        # Create emission prob:
        self.emission_d = {}
        for k, v in pos_to_word_d.items():
            self.emission_d[k] = {}
            for k_inner, v_inner in v.items():
                self.emission_d[k][k_inner] = v_inner/self.pos_total_counts_d[k]

        # Format emission d as wanted:
        self.emission_d_formatted = {}
        for k, v in self.emission_d.items():
            for k_inner, v_inner in v.items():
                key_str = str(k) + ", " + str(k_inner)
                self.emission_d_formatted[key_str] = self.emission_d[k][k_inner]

    def write_hmm(self):
        #Consolidate emission/transition:
        e_t_results_d = {}
        e_t_results_d["transition"] = self.transition_d_formatted
        e_t_results_d["emission"] = self.emission_d_formatted

        # Write the Emission/Transition Prob to a file:
        with open('hmm.json', 'w') as f:
            json.dump(e_t_results_d, f)
    
    def print_results_2(self):
        print("Task 2 Results:")
        print("# of Transition Parameters:", len(self.transition_d_formatted))
        print("# of Emission Parameters:", len(self.emission_d_formatted))
        print("\n")

    def preliminary_greedy(self):
        # Calculate best odds to be t(s1):
        most_likely_start_d = {}
        pos_tags_start = self.train_df[self.train_df["index"] == 1]["pos_tag"]
        for pos_tag in pos_tags_start:
            if pos_tag in most_likely_start_d:
                most_likely_start_d[pos_tag] += 1
            else:
                most_likely_start_d[pos_tag] = 1

        self.most_likely_start_prob_d = {k:v/len(pos_tags_start) for k,v in most_likely_start_d.items()}

    @staticmethod
    def greedy_hmm(df, unknown_word_lst, cnt_d_sorted, emission_d, most_likely_start_prob_d, transition_d):
        """Allows for a greedy implementation of HMM. Input - Dataframe that has words as column."""
        pred_lst = []
        all_words = df["word"].values
        prev_pos = "None"
        for i, word in enumerate(all_words):
            best_prob = 0
            best_pos = "None"
            if (word in unknown_word_lst) or (word not in cnt_d_sorted):
                word = "< unk >"
            if i == 0:
                for pos_tag in emission_d.keys():
                    if word in emission_d[pos_tag]: 
                        e_x1_s1 = emission_d[pos_tag][word]*most_likely_start_prob_d[pos_tag]
                        if e_x1_s1 > best_prob:
                            best_prob = e_x1_s1
                            best_pos = pos_tag
                    else:
                        continue
                pred_lst.append(best_pos)
                prev_pos = best_pos
            else:
                for pos_tag in emission_d.keys():
                    if word in emission_d[pos_tag]:
                        e_x2_s2 = emission_d[pos_tag][word]
                        if pos_tag in transition_d[prev_pos]:
                            t_s2_s1 = transition_d[prev_pos][pos_tag]
                        else:
                            t_s2_s1 = 0
                        prob_ = e_x2_s2*t_s2_s1
                        if prob_ > best_prob:
                            best_prob = prob_
                            best_pos = pos_tag
                    else:
                        continue

                if best_pos == "None": #Edge case where word and POS DO NOT Allign
                    for pos_tag in emission_d.keys():
                        if word in emission_d[pos_tag]:
                            e_x2_s2 = emission_d[pos_tag][word]
                            if e_x2_s2 > best_prob:
                                best_prob = e_x2_s2
                                best_pos = pos_tag
                pred_lst.append(best_pos)
                prev_pos = best_pos
                
        return pred_lst

    def call_greedy_dev(self):
        self.greedy_dev_pred_lst = self.greedy_hmm(self.dev_df, self.unknown_word_lst, self.cnt_d_sorted,
                                        self.emission_d, self.most_likely_start_prob_d, self.transition_d)
        dev_acc = sum(np.array(self.greedy_dev_pred_lst) == self.dev_df["pos_tag"].values)/len(np.array(self.greedy_dev_pred_lst))
        print("Greedy Dev Accuracy: ", dev_acc)
        print("\n")

    def call_greedy_test(self):
        self.greedy_test_pred_lst = self.greedy_hmm(self.test_df, self.unknown_word_lst, self.cnt_d_sorted,
                                        self.emission_d, self.most_likely_start_prob_d, self.transition_d)
        with open("greedy.out", "w") as g:
            test_idx = self.test_df["index"].values
            test_word = self.test_df["word"].values
            for i, pred in enumerate(self.greedy_test_pred_lst):
                g.write(str(test_idx[i]))
                g.write("\t")
                g.write(str(test_word[i]))
                g.write("\t")
                g.write(str(pred))
                g.write("\n")

    @staticmethod
    def viterbi_hmm(df, unknown_word_lst, cnt_d_sorted, emission_d, most_likely_start_prob_d, transition_d):
        """Allows for a viterbi decoding implementation of HMM. Input - Dataframe that has words as column."""
        all_words = df["word"].values
        states = [k for k in emission_d.keys()]
        V = []
        for i, word in enumerate(all_words):
            V.append({})
            current_map = {}
            if (word in unknown_word_lst) or (word not in cnt_d_sorted):
                word = "< unk >"
            if i == 0:
                for pos_tag in states:
                    if word in emission_d[pos_tag]: 
                        e_x1_s1 = np.log(emission_d[pos_tag][word]) + np.log(most_likely_start_prob_d[pos_tag])
                        current_map[pos_tag] = {"prob": e_x1_s1, "prev_state": None}
                    else:
                        current_map[pos_tag] = {"prob": -50, "prev_state": None}
                V[i] = current_map
            else:
                for pos_tag in states:
                    if pos_tag in transition_d[states[0]]:
                        best_prob = V[i-1][states[0]]["prob"] + np.log(transition_d[states[0]][pos_tag])
                    else:
                        best_prob = -1000000
                    past_st = states[0]
                    for prev_tag in states[1:]:
                        if pos_tag in transition_d[prev_tag]:
                            t_s2_s1 = np.log(transition_d[prev_tag][pos_tag])
                        else:
                            t_s2_s1 = -100
                        tr_cost = V[i-1][prev_tag]["prob"] + t_s2_s1 #log or -100 penalty
                        if tr_cost > best_prob:
                            best_prob = tr_cost
                            past_st = prev_tag
                    
                    if word in emission_d[pos_tag]:
                        e_x1_s1 = np.log(emission_d[pos_tag][word])
                    else:
                        e_x1_s1 = -100
                    best_prob = best_prob + e_x1_s1  #log or -100 penalty
                    V[i][pos_tag] = {"prob": best_prob, "prev_state": past_st}
                    
        pred_lst = []
        final_state = "." #init a guess of . ending
        best_prob = -np.inf #init a guess of best prob
        
        #Figure out what the best final state is:
        for st, d_ in V[len(all_words)-1].items():
            if d_["prob"] > best_prob:
                final_state = st
                best_prob = d_["prob"]
                
        #Follow path from end to beginning      
        for i in range(len(all_words)-2, -1, -1):
            pred_lst.append(V[i+1][final_state]["prev_state"])
            final_state = V[i+1][final_state]["prev_state"]
            
        return np.flip(np.array(pred_lst),axis=0)

    def call_viterbi_dev(self):
        self.viterbi_dev_pred_lst = self.viterbi_hmm(self.dev_df, self.unknown_word_lst, self.cnt_d_sorted,
                                        self.emission_d, self.most_likely_start_prob_d, self.transition_d)
        dev_acc = sum(1 for x,y in zip(self.viterbi_dev_pred_lst, self.dev_df["pos_tag"].values) if x == y)/len(self.viterbi_dev_pred_lst)
        print("Viterbi Dev Accuracy: ", dev_acc)
    
    def call_viterbi_test(self):
        self.viterbi_test_pred_lst = self.viterbi_hmm(self.test_df, self.unknown_word_lst, self.cnt_d_sorted,
                                        self.emission_d, self.most_likely_start_prob_d, self.transition_d)
        
        # Write output of Test Data:
        with open("viterbi.out", "w") as v:
            test_idx = self.test_df["index"].values
            test_word = self.test_df["word"].values
            for i, pred in enumerate(self.viterbi_test_pred_lst):
                v.write(str(test_idx[i]))
                v.write("\t")
                v.write(str(test_word[i]))
                v.write("\t")
                v.write(str(pred))
                v.write("\n")

warnings.filterwarnings("ignore")
print("Python version:", python_version())

if len(sys.argv) == 1:
    raise Exception("Must provide data filepath as input.")
else:
    fp = sys.argv[1]


hw_class = run_hw(fp)
#Task 1:
hw_class.read_data()
hw_class.clean_data()
hw_class.get_word_count()
hw_class.create_unknown_key()
hw_class.sort_count_d()
hw_class.write_vocab()
hw_class.print_results_1()

#Task 2:
hw_class.create_transition_prob()
hw_class.create_emission_prob()
hw_class.write_hmm()
hw_class.print_results_2()

#Task 3:
hw_class.preliminary_greedy()
hw_class.call_greedy_dev()
hw_class.call_greedy_test()
hw_class.call_viterbi_dev()
hw_class.call_viterbi_test()

#Task 4:
