#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: Shivam Thakur(spthakur),Sanket Patole(sspatole),Taj Tanveer Shaikh(Taj shaik)
# (based on skeleton code by D. Crandall, Oct 2018)
#

#Approach:-
#The three main parts to this program is calculating the initial probabilites, transition probabilities and the emission probabilities.
#Calculating initial probabilites: I take a dictionary called init_prob for holding the initial probabilities. Initially, I assign an initial probability of 0 to all the training characters.
#Then I take a dictionary called transit_prob to hold the transition probabilities. This is a nested dictionary. The keys for this dictionary are all the training characters and the
#values for each key are also all the training characters. A key-value pair denotes each transition. For example: {'A':{'D':0.06}} denotes that the transition probability of 'A' to
#'Z' is 0.06.
#Then, I read the testing file line by line. For each line, I increment the value of the key of the init_prob dictionary for the first character of the line. I also maintain a count
#variable which counts the total number of lines in the file. Later, I iterate over the init_prob dictionary and divide each value by the count value. Hence, the initial probability
#of each letter is calculated by counting the number of lines which begin with that letter divided by the total number of lines in the test file. Finally, I take the negative log of
#this probability and store it in the dictionary.
#Calculating transition probabilities: For each line, I take 2 variables letter1 and letter2 such that letter2 proceeds letter1 in every iteration. I store the number of times letter2
#proceeds letter1 in the transit_prob dictionary, i.e., transit_prob[letter1][letter2]. I also store the number of times letter1 precedes any other letter in a dictionary called as 
#transitsum. For example, transitsum['A']=13 denotes that letter 'A' precedes any other letter 13 times in the test file. I then iterate over all the possible transitions over the
#training letters and calculate the transition probabilites as transit_prob[letter1][letter2]/transitsum[letter1]. This means that I divide the number of times letter1 precedes
#letter2 by the total number of times letter1 precedes any letter. I then take the negative log of this probability and store it in the dictionary.
#Calculating emission probabilites: I create a dictionary called as emission_prob whose keys are the indeces of the letters in the test image and the values of each key are all the
#training letters. emission_prob[2]['B']=0.5 denotes that the 2nd letter of the test image has an emission probability of 0.5 for the letter 'B'. To calculate this probability, I
#create a function called count() which calculates the number of matching and unmatching characters in the image representation of the test letter and the training letter. I call 
#this function for each test letter and iterate over all the training letters. I take a tuning parameter of 0.3 and calculate the emission probability as (1-t)^m * t^n, where t is 
#the tuning parameter, m is the number of matched characters and n is the number of unmatched characters. Finally, I take the negative log of this probability and store it in the 
#dictionary.
#Then we come to the final part of the program which is implementing the Viterbi algorithm. From this algorithm, we know that the probability of a test letter being a particular
#character is the emission probability of that character multiplied by the max value of product of the previous letter probability and the transition probability from the previous
#letter to the current letter. Since we have to be able to backtrack after we reach the last letter, I created a complex data structure called prob. prob is a dictionary whose keys 
#the indices of the test letters. The values of each key are also dictionaries, whose keys are all the training letters. The value of each nested dictionary is a list which holds the
#maximum value of the prduct of the previous letter probability and the transition probability from the previous the letter to the current letter and also the letter for which this 
#probability is maximum. For example: {2:{'d':[0.03,'E']}} denotes that for the third test letter, the probability that it is 'd' is 0.03 and this probability is maximum if the 
#previous character is 'E'. This way, it becomes easy for us to backtrack. After populating this dictionary for all the test letters, I backtrack by choosing the last letter as the 
#letter with the highest probability and it's previous letter as the letter for which this probability is the highest and so on until I reach the first letter.

#Design decisions:-
#For a particular training letter, if there is no line which starts with that letter in the training file, I take its probability to be 0.0000001.
#For each possible transition i to j, if there is no such transition in the training file, I take the transition probability of i to j to be 0.0000001.

#References:-
#1: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25

init_prob={}
transit_prob={}

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    for i in TRAIN_LETTERS:
        init_prob[i]=0
        transit_prob[i]={}
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

#This holds the number of times a particular character precedes other characters.
transitsum={}

sum=0  #This holds the total number of lines in the training file.
train_file=open(train_txt_fname)  #This is the file handle of the training file
for line in train_file:
	for i in range(0,len(line)-1):
		letter1=line[i]
		if i==0:
			if letter1 in init_prob.keys():
				init_prob[letter1]+=1
				sum+=1
		for j in range(1,len(line)):  #j always occurs after i
			letter2=line[j]
			if letter1 in transit_prob.keys() and letter2 in transit_prob[letter1].keys():
				transit_prob[letter1][letter2]+=1
			elif letter1 in transit_prob.keys() and letter2 not in transit_prob[letter1].keys():
				transit_prob[letter1][letter2]=1
			if i in transitsum.keys():
				transitsum[i]+=1
			else:
				transitsum[i]=1

for k in init_prob.keys():
	init_prob[k]/=sum
	if init_prob[k]==0:
		init_prob[k]=-math.log(0.0000001)
	else:
		init_prob[k]=-math.log(init_prob[k])
				

for i in train_letters:
	for j in train_letters:
		if j in transit_prob[i].keys():
			if i in transitsum.keys():
				transit_prob[i][j]=-math.log(transit_prob[i][j]/transitsum[i])
			else:
				transit_prob[i][j]=-math.log(0.0000001)
		else:
			transit_prob[i][j]=-math.log(0.0000001)


def count(a,b):  #a and b are the image representations of the test letter and train letter respectively
	same=0  #holds the number of matching characters
	diff=0  #holds the number of unmatching characters
	for a1,b1 in zip(a,b):
		for l1,l2 in zip(a1,b1):
			if l1==l2:
				same+=1
			else:
				diff+=1
	return (same,diff)

emission_prob={}  #holds the emission probabilities

for i in range(0,len(test_letters)):
	emission_prob[i]={}
	for letter in train_letters:
		(same,diff)=count(test_letters[i],train_letters[letter])
		prob=pow((0.7),same)*pow(0.3,diff)  #0.3 is the tuning parameter
		emission_prob[i][letter]=-math.log(prob)

prob={}  #holds the final probabilities, i.e., Vi(t).

for i in range(0,len(test_letters)):
	prob[i]={}
	if i==0:
		for j in train_letters:
			prob[i][j]=init_prob[j]+emission_prob[i][j],j  #For the first testing letter, we give the previous letter for which the current letter has the maximum probability as the current letter itself, as there is no previous letter since this is the first letter of the test case.
	else:
		for j in train_letters:
			temp={}  #This holds the product of the previous probability of a letter and the transition probability from the previous letter to the current letter for all the training letters.
			for k in prob[i-1].keys():
				temp[k]=prob[i-1][k][0]+transit_prob[k][j]
			minkey=min(temp,key=temp.get)  #This function chooses the key which has the minimum value. The logic for this function is taken from reference #1. So minkey holds the previous letter for which the probability will be the highest for the current letter.
			minval=min(temp.values())  #minkey holds the max value of the probability, i.e., max(Vi-1(t)*Pij) where Vi-1(t) is the probability of the previous letter and Pij is the transition probability from the previous letter to the current letter.
			prob[i][j]=emission_prob[i][j]+minval,minkey


N=len(test_letters)-1
a=[0]*(N+1)

for i in range(0,len(test_letters)):
	a[i]=min(emission_prob[i],key=emission_prob[i].get)  #a[i] stores the letter for which the emission proobability is the highest for the ith test character.
print('Simple: '+''.join(map(str,a)))


i=N
while i>=0:
	if i==N:
		a[i]=min(prob[i],key=prob[i].get)
		maxval=a[i]
	else:
		maxkey=prob[i+1][maxval][1]
		a[i]=maxkey
		maxval=a[i]
	i-=1


print('Viterbi: '+''.join(map(str,a)))

print('Final answer:\n'+''.join(map(str,a)))
