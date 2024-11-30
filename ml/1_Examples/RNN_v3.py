################################################################################################################################
# This code tries to replicate Karpathy's Minimal Character level RNN (https://gist.github.com/karpathy/d4dee566867f8291f086)
# Text file: Three Little Pigs short story (selected for its smaller word-vocabulary and smaller overall size.)
# Network Architechture: 3 Layer RNN that reads 20 characters at a time (timeSteps)
# Generating Text: Uses the trained RNN and a starting character to generate 100 characters one after the other.
# Result: Despite generating non-sense text, we see structured words being generated that make sense in immediate neighbourhood

#### at Loss 65, Generated Text: hzvpyibielhknyonubhmblfeiih ystsvjjkpkfvbmaolmtrewzbtrmkeuanhbbfpeaznhuz budgyaeryfatpbxvalalavxumayjviprmwpwaspsanpauuxifglgvaxdhrk vntsshaidwfimwpxo 
#### at Loss 55, Generated Text: the cbafclulf and rot thog san soo shhe monnleryer and hhe soeft icpic dot and ay ont ir wet se xut yui pok be let he cofiktese ta cotid time the cid
#### at Loss 45, Generated Text: epan so and me po thet tald pag anw moilf and so houpig anghrem ror the woll doand piy gill se shing o the wat wenln riutt and thatt to be i the horen
#### at Loss 35, Generated Text: id oly pfing dhat afy get berand chuck nell and or huffod and at that tothe lhathe wat fou lind tho wat bapbece the come indyered andence to soreh i h
#### at Loss 25, Generated Text: voust shit thett a lime and thin the pig you tt and it come and you hid sougot up i to pave boing mhoar the howse the little pig i comf thuvery dorne
#### at Loss 15, Generated Text: d buf he hair and little pig i wull thoff comong n the man kat you downan wilf a poicen you hy bome the pig an gi sore with at buffad oll his when a t
#### at Loss 10, Generated Text: h the man did and the you go yome with so the mour tor and said little pig the secoceply cas aw so fich ffus hover and was hus he har sase with me tor
#### at Loss 6, Generated Text: the ther come thaw wherer the wolf brand that hi ko come hes ald sat so build a house wnice pufnin where thered the little pig but onto pld buttle pu

# Source article: https://karpathy.github.io/2015/05/21/rnn-effectiveness/
################################################################################################################################

# Load dependencies
import torch as t
import string
from random import choices

# Load Text
PATH = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/character_level_rnn/three_little_pigs.txt'
sampleText = open(PATH, 'r').read()

# Make lowercase
sampleText = sampleText.lower()

# Remove non-ascii
for i in sampleText:
    if i not in string.ascii_lowercase + ' ' + '\n':
        sampleText = sampleText.replace(i, '')

# Remove Whitespaces
sampleText = sampleText.replace('\n', ' ')
sampleText = ' '.join(sampleText.split())

# Make text simpler
# sampleText = sampleText.replace(' pigs', ' pig')
# sampleText = sampleText.replace(' ill', ' i')
# sampleText = sampleText.replace(' chinny', ' chin')
# sampleText = sampleText.replace(' huffed', ' huff')
# sampleText = sampleText.replace(' puffed', ' puffed')

# Remove Stop Words
# from nltk.corpus import stopwords
# stopwords = set(stopwords.words('english'))
# for i in sampleText:
#    if i in stopwords:
#        sampleText = sampleText.replace(' ' + i + ' ', ' ')

# Remove Low Frequency Words
# words = sampleText.split()
# for i in words:
#    if words.count(i) < 3:
#        sampleText = sampleText.replace(' ' + i + ' ', ' ')

# Vocab Size
uniqueChars = list(set(sampleText))
uniqueChars.sort()
noOfUniqueChars = len(uniqueChars)

# Index - Character Mapping
idxToChar = {i: j for i, j in enumerate(uniqueChars)}
charToIdx = {j: i for i, j in enumerate(uniqueChars)}

# Character To One Hot Encoded Tensor
def charToTensor(char, length=noOfUniqueChars):
    x = t.zeros(1, length)
    x[:, charToIdx[char]] = 1
    return x

# 3-Layer RNN
hiddenNeurons1 = 100
hiddenNeurons2 = 75
hiddenNeurons3 = 50
inputNeurons = noOfUniqueChars
outputNeurons = noOfUniqueChars
learningRate = 0.001
timeSteps = 20

# Initalize Parameters
initHidden1 = t.zeros(1, hiddenNeurons1)
initHidden2 = t.zeros(1, hiddenNeurons2)
initHidden3 = t.zeros(1, hiddenNeurons3)
Wh1 = t.tensor(t.randn(inputNeurons + hiddenNeurons1, hiddenNeurons1) * 0.01, requires_grad=True)
Wh2 = t.tensor(t.randn(hiddenNeurons1 + hiddenNeurons2, hiddenNeurons2) * 0.01, requires_grad=True)
Wh3 = t.tensor(t.randn(hiddenNeurons2 + hiddenNeurons3, hiddenNeurons3) * 0.01, requires_grad=True)
Wy = t.tensor(t.randn(hiddenNeurons3, outputNeurons) * 0.01, requires_grad=True)
Bh1 = t.zeros(1, hiddenNeurons1, requires_grad=True)
Bh2 = t.zeros(1, hiddenNeurons2, requires_grad=True)
Bh3 = t.zeros(1, hiddenNeurons3, requires_grad=True)
By = t.zeros(1, outputNeurons, requires_grad=True)

# Custom Cross Entropy Loss Function, had some difficulties with t.nn.CrossEntropyLoss()
def crossEntropyLoss(yprob, ytrue):
    return - t.mm(t.log(yprob), ytrue.T)

# ADAM Optimizer
optimizer = t.optim.Adam([Wh1, Wh2, Wy, Bh1, Bh2, By], lr=learningRate)

# Perform 1 Training Iteration on Single Text Sequence
def train(Text):
    inputText = Text[:-1]
    labelText = Text[1:]
    loss = 0
    h_1_prev, h_2_prev, h_3_prev = initHidden1, initHidden2, initHidden3
    for i in range(timeSteps):
        x = charToTensor(inputText[i])
        y = charToTensor(labelText[i])
        h_1_next = t.nn.Tanh()(t.mm(t.cat((x, h_1_prev), dim=1), Wh1) + Bh1)
        h_2_next = t.nn.Tanh()(t.mm(t.cat((h_1_next, h_2_prev), dim=1), Wh2) + Bh2)
        h_3_next = t.nn.Tanh()(t.mm(t.cat((h_2_next, h_3_prev), dim=1), Wh3) + Bh3)
        yhat = t.mm(h_3_next, Wy) + By
        yprob = t.nn.Softmax(dim=1)(yhat)
        loss += crossEntropyLoss(yprob, y)
        h_1_prev, h_2_prev, h_3_prev = h_1_next, h_2_next, h_3_next
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# Test Training
# for i in range(100):
#    train(sampleText[0:timeSteps + 1])

# Scoring/Sample Function
# Uses Softmax Probabilities to obtain next Character
def GenerateText(initChar, timeSteps=timeSteps):
    x = charToTensor(initChar)
    predictedCharacters = []
    h_1_prev, h_2_prev, h_3_prev = initHidden1, initHidden2, initHidden3
    for step in range(timeSteps):
        h_1_next = t.nn.Tanh()(t.mm(t.cat((x, h_1_prev), dim=1), Wh1) + Bh1)
        h_2_next = t.nn.Tanh()(t.mm(t.cat((h_1_next, h_2_prev), dim=1), Wh2) + Bh2)
        h_3_next = t.nn.Tanh()(t.mm(t.cat((h_2_next, h_3_prev), dim=1), Wh3) + Bh3)
        yhat = t.mm(h_3_next, Wy) + By
        yprob = t.nn.Softmax(dim=1)(yhat)
        h_1_prev, h_2_prev, h_3_prev = h_1_next, h_2_next, h_3_next
        chosenChar = choices(uniqueChars, weights=yprob.reshape(-1))[0]
        x = charToTensor(chosenChar)
        predictedCharacters.append(chosenChar)
    return ''.join(predictedCharacters)

# Cycle through the SampleText jumping timeSteps Characters at a time
# Train single iteration and update weights
# At end of SampleText, revert to the start and begin again
# Every 100 Training Iteration Generate Text from first character in sequence

start, end = 0, timeSteps + 1
epoch, iteration = 0, 0
globalLoss = - t.log(t.tensor(1 / noOfUniqueChars)) * timeSteps
while True:
    if end > len(sampleText):
        start, end = 0, timeSteps + 1
        epoch += 1

    TextSnippet = sampleText[start:end]
    iterationLoss = train(TextSnippet)
    globalLoss = 0.9999 * globalLoss + 0.0001 * iterationLoss

    # print(f'Epoch: {epoch}, Iteration: {iteration}, Global Loss: {globalLoss.item()}')
    # print(f'Start: {start}, End: {end}, {TextSnippet}')

    if iteration % 1000 == 0:
        print(f'\nGlobal Loss: {globalLoss.item()}')
        print(f'Generated Text: {GenerateText(choices(uniqueChars)[0], timeSteps=150)}')

    iteration += 1
    start += timeSteps
    end += timeSteps
