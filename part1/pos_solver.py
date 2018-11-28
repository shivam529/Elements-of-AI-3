#!/usr/bin/env python
###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
##Shivam Thakur(spthakur),Sanket Patole(sspatole),Taj tanveer shaikh(tajsahik)
# (Based on skeleton code by D. Crandall)
#
#
####
# We kept our intial,emision,transition, and  individual probabilities in dictionaries,they are:
# Transiton:P(Si+1|Si) Line 23
# Emission: P(Wi|Si) Line 25
# Initial_prob: P(S1) Line 24
# Individual: P(Si) Line 26  *Note: There's a typo in dictionary name for individual probabilities,it's named "noun_prob"*
# Transition2: P(Si|Si-1,Si-2) Line 27 *Note: Used in MCMC since the Bayes' nets (figure 1c) reqires so*
####

import random
import math
import numpy as np
class Solver:
    
    c=0.0
    transition=dict()
    initial_prob=dict()
    emission=dict()
    noun_prob=dict()
    transition2=dict()
    pos_tags=['adj','adv','adp','conj','det', 'noun', 'num', 'pron', 'prt', 'verb','x','.'] ## Fixed list of POS tags used to iterate through in later stages
    ## Calculate the log of the posterior probability of a given sentence##
    ## This function caluclates the posterior probability of Tags|words for each model and hence the probabilities are caculated according to##
    ## the respective Bayes' Nets##
    def posterior(self, model, sentence, label):
        
        if model == "Simple":
            pos=0.0
            for i in range(len(label)):
                pos+=math.log(self.emission.get((sentence[i],label[i]),0.000000000000001))+math.log(self.noun_prob.get(label[i]))
            return pos
        elif model == "Complex":
            pos=0.0
            for i in range(len(label)):
                if i==0:
                        pos+=math.log(self.initial_prob.get(label[i])*(self.emission.get((sentence[i],label[i]),0.000000000000001)))
                elif i==1:
                     pos+=math.log(self.emission.get((sentence[i],label[i]),0.00000000000001)*self.transition.get((label[i],label[i-1]),0.000000000000001))

                else:
                        pos+=math.log(self.emission.get((sentence[i],label[i]),0.00000000000001)*self.transition.get((label[i],label[i-1]),0.000000000000001)*self.transition2.get((label[i],(label[i-1],label[i-2])),0.0000000000001))
                     
            return pos
        elif model == "HMM":
            pos=0.0
            for i in range(len(label)):
                if i==0:
                    pos+=math.log(self.emission.get((sentence[i],label[i]),0.0000000000001))+math.log(self.initial_prob.get(label[i]))
                else:
                    pos+=math.log(self.emission.get((sentence[i],label[i]),0.0000000000001))+math.log(self.transition.get((label[i],label[i-1]),0.0000000000000001))
            return pos
        else:
            print("Unknown algo!")

    ##### TRAINING ######
    #####Finding the required probabilities here#####
    def train(self, data):
        
        ## P(S1) (Initial Probs) * Dictionary consists of key:values as (S1):probabalities *
        for line in data:
            self.initial_prob[line[1][0]]=self.initial_prob.get(line[1][0],0)+1
        total_init=float(sum(self.initial_prob.values()))
        self.initial_prob={k:v/total_init for (k,v) in self.initial_prob.items()}
        
         ##P(Si+1|Si) (Transition Probs) * Dictionary consists of key:values as (Si+1,Si)tuple:probabilities *
        for line in data:
            for s in range(len(line[1])-1):
                self.transition[(line[1][s+1],line[1][s])]=self.transition.get((line[1][s+1],line[1][s]),0)+1
        total_tran=dict() # total no. of cases where transition from Si occurs

        for states,v in self.transition.items():
            total_tran[states[1]]=total_tran.get(states[1],0)+v
        self.transition={k:v/float(total_tran[k[1]]) for (k,v) in self.transition.items()}
        # print("tr",self.transition)

        ## P(Si) (Individual states Probs)
        for line in data:
         for s in range(len(line[1])):
            self.noun_prob[line[1][s]]=self.noun_prob.get(line[1][s],0)+1
        noun_count=self.noun_prob
        t=float(sum(noun_count.values()))
        self.noun_prob={k:v/t for (k,v) in self.noun_prob.items()}
    
        ##P(Wi|Si) (EMMISSION PROBABILITIES)
        for line in data:
            for s in range(len(line[1])):
                self.emission[(line[0][s],line[1][s])]=self.emission.get((line[0][s],line[1][s]),0)+1
        self.emission={k:v/float(noun_count[k[1]]) for (k,v) in self.emission.items()}


        ##P(Si+2|Si+1,Si) (TRANSITION PROBABILITIIES for states given previous two states)
        for line in data:
            for s in range(len(line[1])-2):
                self.transition2[(line[1][s+2],(line[1][s+1],line[1][s]))]=self.transition2.get((line[1][s+2],(line[1][s+1],line[1][s])),0)+1
        den=dict()
        for line in data:
            for s in range(len(line[1])-1):
                den[(line[1][s+1],line[1][s])]=den.get((line[1][s+1],line[1][s]),0)+1 ## Total No. of cases where transition from Si+1,Si occurs

        
        self.transition2={o:p/float(den.get(o[1],999999999999)) for (o,p) in self.transition2.items()}
       
    ## Simplified uses figure 1b from the assignment and maximimzes over all possibilites of POS tags for each word,maximxing on each on of them##
    ## Hence, probabilities used here are Indivialual state probabilities(Noun_prob)*typo*, and the emission probabilities for word|state from ##
    ## emission dictionary and if a given emission probability is not found, we assing it a very small value of 0.00000000000000001##
    def simplified(self, sentence):
        sequence=[]
        final=[]
        for word in sentence:
             sequence=[self.emission.get((word,state),0.00000000000000001)*self.noun_prob.get(state) for state in self.pos_tags]
             index=sequence.index(max(sequence))
             final.append(self.pos_tags[index])
        return final

    ##MCMC uses figure 1c using gibbs sampling where we start with initial pos tags noun for all the words in the sentence and the find 12 possoble tags ##
    ## for each word in the sentence GIVEN the rest of the states(i.e, keeping rest of the states fixed) and assign a tag to the current state randomly.##
    ## We discarded the first 100 iterations as a warm period and henceforth check for convergence by checking the last five POS tags List and if the last##
    ## five Lists of POS tags generated are same, we deem it converged and break, hence sending the corresponding Tags to labels.py for each senetence.##
    def complex_mcmc(self, sentence):
        intial_tags=[ "noun"]*len(sentence)
        check=[]
        for iterations in range(200):
            final_tags=[]
            
            for words in range(len(sentence)):
                temp_tags=[]
                for i in  range(len(self.pos_tags)):
                    if words==0: ## First state
                        temp_tags.append(self.initial_prob.get(self.pos_tags[i])*self.emission.get((sentence[words],self.pos_tags[i]),0.00000000000000001))
                    elif words==len(sentence)-1: ## Last state wont have P(Si+1|Si) and P(Si+2|Si+1,Si), hence special case##
                        temp_tags.append(self.emission.get((sentence[words],self.pos_tags[i]),0.00000000000000001)*self.transition.get((self.pos_tags[i],intial_tags[words-1]),0.00000000000000001)*self.transition2.get((self.pos_tags[i],(intial_tags[words-1],intial_tags[words-2])),0.00000000000000001))
                    elif words==len(sentence)-2: ## Special case again, wont have P(Si+2|Si+1,Si)
                        temp_tags.append(self.emission.get((sentence[words],self.pos_tags[i]),0.00000000000000001)*self.transition.get((self.pos_tags[i],intial_tags[words-1]),0.00000000000000001)*self.transition2.get((self.pos_tags[i],(intial_tags[words-1],intial_tags[words-2])),0.00000000000000001)*self.transition.get((intial_tags[words+1],self.pos_tags[i]),0.00000000000000001))
                    # elif words==1:
                    #     temp_tags.append(self.emission.get((sentence[words],self.pos_tags[i]),0.00000000000000001)*self.transition.get((self.pos_tags[i],intial_tags[words-1]),0.00000000000000001)*self.transition2.get((intial_tags[words+2],(intial_tags[words+1],self.pos_tags[i])),0.00000000000000001)*self.transition.get((intial_tags[words+1],self.pos_tags[i]),0.00000000000000001))

                    else: ## All the middle states
                        temp_tags.append(self.emission.get((sentence[words],self.pos_tags[i]),0.00000000000000001)*self.transition.get((self.pos_tags[i],intial_tags[words-1]),0.00000000000000001)*self.transition2.get((self.pos_tags[i],(intial_tags[words-1],intial_tags[words-2])),0.00000000000000001)*self.transition2.get((intial_tags[words+2],(intial_tags[words+1],self.pos_tags[i])),0.00000000000000001)*self.transition.get((intial_tags[words+1],self.pos_tags[i]),0.00000000000000001))
                     


                        
                norm=[item/sum(temp_tags) for item in temp_tags] ##normalizing the probabilities##
                tag=np.random.choice(['adj','adv','adp','conj','det', 'noun', 'num', 'pron', 'prt', 'verb','x','.'], 1, p=norm)
                final_tags.append(tag[0])
            intial_tags=final_tags
            check.append(final_tags) ## list of tags for every sentence through every iteration to be checked for convergence using this list
            if(iterations>50): ## skipping first 100 iterations and then checkin for convergence
                ctr=0
                for items in check[-5:]:
                    if final_tags==items:
                        ctr+=1
                if(ctr>=4): ## if last list of tags are same, its converged so break##
                    
                    
                    break
        return intial_tags
       
    ## Uses figure 1a, We have created a numpy matrix to populate values going state by state and then finding the minimum of negative log of all the##
    ## probabilities in track dictionary and go back ward to find where each state came from.
    def hmm_viterbi(self, sentence):
        viterbi_array=np.zeros((len(self.pos_tags),len(sentence)))
        ret=[]
        final=[]
        j=0
        for word in sentence:
            track=dict()
            i=0
            if(j==0):
                for state in self.pos_tags:
                    viterbi_array[i,j]=-math.log(self.emission.get((word,state),0.00000000000000001))-math.log(self.initial_prob.get(state))
                    i+=1
            else:
                for state in self.pos_tags:
                    emission_temp=-math.log(self.emission.get((word,state),0.00000000000000001))
                    mininmized=[viterbi_array[iteration][j-1]-math.log(self.transition.get((state,states),0.00000000000000001)) for iteration,states in enumerate(self.pos_tags,0)]
                    track[state]=self.pos_tags[mininmized.index(min(mininmized))]
                    max_prob=min(mininmized)
                    viterbi_array[i,j]=emission_temp+max_prob
                    i=i+1
            if(j!=0):
                final.append(track)
            j+=1
                
        c=np.argmin(viterbi_array,axis=0)
        c=c[-1]
        temp_state=self.pos_tags[c]
        ret.append(temp_state)
        final.reverse()

        for elem in final:
            for keys,values in elem.items(): 
                if temp_state==keys:
                    ret.append(values)
                    temp_state=values
                    break
        ret.reverse()
        return ret
    
    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
        
