---
layout: post
title: "Boosting: AdaBoost & Gradient Boosting"
comments: true
description: "Boosting: AdaBoost & Gradient Boosting"
keywords: "dummy content"

---

이번 포스팅에서는 앙상블 기법 중 Boosting에 해당하는 AdaBoost와 Gradient Boosting Machine에 대해 살펴보겠습니다.
해당 포스팅은 **[고려대학교 산업경영공학과 강필성 교수님](https://github.com/pilsung-kang)** 강의를 참고했음을 밝힙니다.

<h1>Ensemble Learning</h1>
<h3>앙상블은 무엇일까요?</h3>
다양한 데이터 분석 기법들은 대부분 단일모형을 가지고 예측 또는 분류를 진행합니다.

**[Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)**은 다양한 단일모형의 조합을 통하여 성능을 끌어 올리기 위하여 고안된 방법입니다.

앙상블을 구성하는 단일모형들은 조합 안에서 결과**(예측값 또는 예측 클래스)**를 피력하고, 최종결과는 단일모형들의 표결에 따라 결정됩니다.

데이터 분석에 이용되는 모형들은 대부분 아래와 같이 표현이 가능합니다.

![그림1](https://imgur.com/1NyreI6.png)

이때 각 객체마다의 MSE는

![그림2](https://imgur.com/MuM8xlg.png)

와 같이 분해가 가능합니다.

따라서 우리는 통제가 가능한 오차인 편향과 분산에 대해 관심을 가질 필요가 있습니다.

편향은 실제값과 추정된 값들의 차이로 설명할 수 있습니다. 그리고 분산은 객체들의 퍼져있는 정도로 설명할 수 있습니다.

이러한 의미로 보았을 때, 아래의 그림에서 왼쪽은 낮은 편향과 높은 분산을 가지는 경우에 해당합니다.

반대로 아래의 그림에서 오른쪽은 높은 편향과 낮은 분산을 가지는 경우입니다.
<h2></h2>
![그림3](https://imgur.com/AP2yxxU.png)
<h2></h2>

편향이 낮고 분산이 높은 모형은 데이터가 변경됨에 따라 성능이 바뀌게 되며 노이즈에 민감하다는 특징을 가지고 있습니다.

이에 비해 편향이 높고 분산이 낮은 모형은 데이터가 변경됨에 따라 성능은 일정하지만 그 값이 실제 값과 많이 다를 수 있다는 특징을 가지고 있습니다.

즉, **전자의 모형**(ex. 의사결정 나무모형, 인공신경망, SVM 등)을 통해 추정된 값들의 평균은 실제 값과 비슷할 수 있으며

**후자의 모형**(ex. 로지스틱 회귀모형, 선형 판별분석 등)은 추정 값들의 평균이 실제 값과 많이 다를 수 있습니다.

이렇듯 다양한 모형들은 다양한 성질을 가지고 있습니다.

앙상블 모형은 이런 모형들의 특성을 반영하여 조금 더 잘 작동하도록 도와주는 역할을 합니다.

<br>


<h3>그렇다면 앙상블 기법을 사용하는 이유는 무엇일까요?</h3>

앙상블 모형을 사용하는 이유는 **이론적으로** 단일모형의 에러보다는 작은 에러를 보장해주기 때문입니다.

여기서 **이론적**이라는 부분을 강조한 이유는 아래의 수식에서 설명드리겠습니다.

복수의 모형들을 잘 조합하여 모형의 **분산을 낮추는 방법을 Bagging**이라고 하며 모형의 **편향을 낮추는 방법을 Boosting**이라고 합니다.

이 포스팅에서는 **편향을 낮추는 방법인 Boosting**에 대해 중점적으로 다룹니다.

우선 개별 모형들은

![그림4](https://imgur.com/ha8GEOG.png)

와 같고 오차는 평균이 0이고 서로 독립이라는 아래와 같은
![그림5](https://imgur.com/cjj33zg.png)

가정을 하게 됩니다. 그러면 **M**개의 각 모형들에 의한 에러의 평균은

![그림6](https://imgur.com/CeJyPxb.png)

와 같고 **M**개의 개별 모형들의 에러들은

![그림7](https://imgur.com/PmHpkcR.png)

로 계산됩니다. 이제 위에서 가정한 식에 의해

![그림8](https://imgur.com/JJaWL4I.png)

가 성립하나 이는 상당히 **이상적**인 가정에 의한 결과입니다.

따라서 이 결과는 코시 부등식에 의해

![그림9](https://imgur.com/yR9FRi7.png)

가 성립합니다. 즉, 앙상블 기법을 사용하게 되면 단일 모형들의 에러보다는 작은 에러가 발생됨을 보장하게 됩니다.

이러한 이유에 의해 앙상블 기법을 사용하게 됩니다.

<br>


<h3>Bagging vs Boosting</h3>

앙상블 기법은 크게 Bagging과 Boosting으로 구분할 수 있습니다.

Bagging은 Bootstrap Aggregating을 의미합니다.

[Bootsrap](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))은 크기가 N인 표본에서 복원추출(With replacement)로 N개의 sub-표본을 추출하는 샘플링 기법을 의미합니다.

Bagging은 B개의 Bootstrap에 대해 모형을 각각 적용하여 최종 결과를 도출하여 예측 또는 분류를 합니다.

이에 비해 Boosting은 이전 모형에서 정확히 예측 또는 분류하지 못한 객체를 더 잘 예측 또는 분류할 수 있도록 모형을 개선합니다.

Bagging과 Boosting의 차이는 아래의 그림과 같습니다.

![그림10](https://imgur.com/JwKlNjt.png)

즉 Bagging은 병렬처리가 가능하며 알고리즘에 독립적인 샘플을 사용하며, Boosting은 Sequential한 방법에 의해 작동합니다.

Bagging은 개별 모형의 분산을 낮추기 위해 사용되기 때문에, **분산이 큰 모형-유연/복잡한 모형-에 적용하는 것이 효과적**입니다.

Boosting은 개별 모형의 편향을 낮추기 위해 사용되기 때문에, **편향이 큰 모형-선형/단순한 모형-에 적용하는 것이 효과적**입니다.

<br>


<h3>Boosting: AdaBoost</h3>

Boosting 중 가장 유명한 AdaBoost는 이전 시점에서 잘 틀리는 부분을 다음 시점에서 더 잘 맞추도록 객체가 선택될 확률을 업데이트합니다.

AdaBoost는 weak 모형들을 주로 사용합니다.

여기서 말하는 weak 모형은 선형에 가까운 모형을 의미합니다.

AdaBoost는 아래의 알고리즘에 의해 작동합니다.

![그림11](https://imgur.com/oqRFe9O.png)

AdaBoost는 각 모형의 오분류율을 계산한 뒤, 오분류율을 이용해 alpha 함수를 만듭니다.

**alpha 함수는 크게 두 가지 역할을 합니다.** 첫 번째는 개별 모형에 대한 가중치이며 두 번째는 객체 선택 확률의 가중치입니다.

이에 대해 구체적으로 알아보겠습니다.

우선 오분류율이 0.5보다 큰 경우는 제외하기 때문에 오분류율의 범위는 [0, 0.5)가 됩니다.

오분류율이 0에 가까워지면 alpha 함수 값은 무한대로 커지게 되며 0.5에 가까워지면 alpha 함수 값은 0에 가까워지게 됩니다.

즉 alpha 함수는 잘 맞추는 모형(== 오분류율이 0에 가까운 모형)에 대해서는 큰 가중치를 주며

잘 맞추지 못하는 모형(== 오분류율이 0.5에 가까운 모형)에 대해서는 0에 가까운 가중치를 주게 됩니다.

그리고 현재 시점 t에서 모형이 정분류인 경우

![그림12](https://imgur.com/9W4o9G3.png)

이므로 해당 객체는 다음 시점에서 선택될 확률이 작아지게 됩니다.

반대로 현재 시점 t에서 모형이 오분류인 경우

![그림13](https://imgur.com/f9bQUvb.png)

로 계산되므로 해당 객체는 다음 시점에서 선택될 확률이 커지게 됩니다.

따라서 alpha에 의해 AdaBoost는 잘 못 맞추는 객체를 맞추는 방향으로 모형을 개선하게 됩니다.

그러므로 Bagging은 Ensemble Learning을 하는 동안 객체가 랜덤하게 선택되지만

**Boosting은 Ensemble Learning을 하는 동안 특정 객체(잘 못 맞추는 객체)가 주로 선택**되게 됩니다.

이러한 특징 때문에 Boosting은 이상치나 노이즈에 민감합니다.

<br>

<h3>AdaBoost in Python</h3>

지금까지 설명한 Adaboost를 **[Python 프로그램](https://www.python.org/)**을 통해 구현해보겠습니다.

<h4>Source code</h4>

**[Python 프로그램](https://www.python.org/)**을 통해 구현한 AdaBoost입니다.

AdaBoost의 Base Learner는 **[Decision Stump](https://en.wikipedia.org/wiki/Decision_stump)**를 사용하였습니다.

아래의 코드는 Decision Stump 관련 코드입니다.

우선 Threshold값을 기준으로 좌우로 나누는 함수입니다.
```python
# Function of Decisioin Stump
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
```

그리고 위의 함수를 사용하여 Best stump를 반환하는 함수에 대한 code 입니다.
```python
# Decision Stump
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst
```
이제 weak model(==base learner)를 받아 Boosting을 진행하는 함수입니다.

중간에 expon 함수를 통해 alpha 함수가 계산되며 그 다음 줄에서 D가 업데이트 되는 것을 보실 수 있습니다.
```python
# Train: AdaBoost
def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    #mis the number of datapoints in a dataset
    m = shape(dataArr)[0]
    #D holds all the weights of each peice of data
    D = mat(ones((m,1))/m)   #init D to all equal
    #aggregrate estimate of the class for every data point
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        print("D:",D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calculate alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)#store Stump Params in Array
        print ("classEst: ",classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon)) #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print ("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print ("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr ,aggClassEst
# Predict: AdaBoost
def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst =stumpClassify(dataMatrix,classifierArr[0][i]['dim'],\
                                 classifierArr[0][i]['thresh'],\
                                 classifierArr[0][i]['ineq'])#call stump classify
        aggClassEst += classifierArr[0][i]['alpha']*classEst
        print (aggClassEst)
    return sign(aggClassEst)
```
실제 데이터를 로딩하여 작동 여부와 결과를 확인해보겠습니다.

데이터는 **[이곳(Training set)](https://github.com/lee-ju/lee-ju.github.io/blob/master/_posts/horseColicTraining2.txt)**과 **[이곳(Test set)](https://github.com/lee-ju/lee-ju.github.io/blob/master/_posts/horseColicTest2.txt)**에서 가져오실 수 있습니다.
```python
# Function: Loading data
from numpy import *
def loadDataSet(fileName): #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
    # Practice: Real data
# Loading a Training and Test data set
# ...부분에 hoseColicTraining2.txt와 horseColicTest2.txt 파일의 위치를 적어주세요!
datArr, labelArr = loadDataSet(".../horseColicTraining2.txt") # Training set
print("Train_Dim: {}".format(shape(datArr)))
testArr, testLabelArr = loadDataSet(".../horseColicTest2.txt") # Test set
print("Test_Dim: {}".format(shape(testArr)))
```
Training set에 대해 AdaBoost를 실시합니다.
```python
# Model fitting
classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
```
Test set에 적용 후, test error raate를 확인한 결과 0.2687임을 알 수 있습니다.

단순한 Decision stump를 사용한 결과치고는 훌륭한 것을 알 수 있습니다.

```python
# Prediction
prediction10 = adaClassify(testArr, classifierArray)
# Checking a Performance
test_m = shape(testArr)[0]
errArr = mat(ones((test_m, 1)))
err = errArr[prediction10 != mat(testLabelArr).T].sum()
err_rate = err / test_m
print("Test error rate: {:.4f}".format(err_rate))
```
```python
[Out]:
Test error rate: 0.2687
```
<h4>Scikit learn code</h4>
**[Python 프로그램](https://www.python.org/)**에서 모델링을 위한 **[scikit learn 라이브러리](https://scikit-learn.org/stable/index.html)**를 사용하여 같은 데이터를 분석해보겠습니다.

이번에는 Base learner를 Decision stump 뿐만 아니라 **[naive bayesian 모형](https://ratsgo.github.io/machine%20learning/2017/05/18/naive/)**도 적용해보도록 하겠습니다.

데이터를 가져오는 코드를 먼저 작성 후
```python
# Function: Loading data
from numpy import *
def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
```
이를 이용해 데이터를 가져옵니다. scikit learn에서는 AdaBoost를 **[sklearn.ensemble.AdaBoostClassifier()](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)**로 제공하고 있습니다.

Decision stump based AdaBoost입니다.
```python
# Practice: Real data
# Loading a Training and Test data set
train_x, train_y = loadDataSet(".../horseColicTraining2.txt") # Training set
print("Train_Dim: {}".format(shape(train_x)))
test_x, test_y = loadDataSet(".../horseColicTest2.txt") # Test set
print("Test_Dim: {}".format(shape(test_x)))
# Importing a Library
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
Tree_BaseLearner = DecisionTreeClassifier(criterion='gini', splitter='best',
                                     max_depth=None, min_samples_split=2,
                                     min_samples_leaf=1, min_weight_fraction_leaf=0,
                                     max_features=None, max_leaf_nodes=None,
                                     min_impurity_decrease=0, min_impurity_split=None,
                                     class_weight=None, presort=False,
                                     random_state=None)
# Tree based AdaBoosting
Tree_AdaBoost = AdaBoostClassifier(base_estimator=Tree_BaseLearner,
                                n_estimators=50, learning_rate=1.0,
                                algorithm='SAMME.R', random_state=None)
                                tab_model = Tree_AdaBoost.fit(train_x, train_y)
train_pred = tab_model.predict(train_x)
train_score = metrics.accuracy_score(train_y, train_pred)
print('train score: {:.4f}'.format(train_score))
test_pred = tab_model.predict(test_x)
test_score = metrics.accuracy_score(test_y, test_pred)
print('test score: {:.4f}'.format(test_score))
#[Out]: train score: 0.9967
#[Out]: test score: 0.6418
```

이전의 Source Code와 비슷한 성능을 보이는 것을 알 수 있습니다.

이번에는 naive bayesian을 base learner로 적용해보면
```python
# Naive Bayesian based learner
from sklearn.naive_bayes import MultinomialNB
NB_BaseLearner = MultinomialNB(alpha=1,
                               fit_prior=True,
                               class_prior=None)
# NB based AdaBoosting
NB_AdaBoost = AdaBoostClassifier(base_estimator=NB_BaseLearner,
                                 n_estimators=50, learning_rate=1.0,
                                 algorithm='SAMME.R', random_state=None)
nbab_model = NB_AdaBoost.fit(train_x, train_y)
train_pred = nbab_model.predict(train_x)
train_score = metrics.accuracy_score(train_y, train_pred)
print('train score: {:.4f}'.format(train_score))
test_pred = nbab_model.predict(test_x)
test_score = metrics.accuracy_score(test_y, test_pred)
print('test score: {:.4f}'.format(test_score))
```
```python
[Out]:
train score: 0.6254
test score: 0.4925
```
Decision stump가 base learner인 경우보다 성능이 현저히 낮은 것을 알 수 있습니다.

따라서 AdaBoost를 사용하는 경우, base learner는 단순한 모형일수록 우수한 성능을 기대할 수 있습니다.

<br>

<h3>Boosting: Gradient Boosting Machine(GBM)</h3>

Gradient Descent는 손실함수를 최소화하는 파라미터를 찾는 최적화 방법이다.

손실함수를 파라미터로 미분해서 기울기를 구하며, 손실함수가 Minimum이 되는 방향으로 움직인다.

Gradient Descent에 대한 자세한 내용은 **[이곳](https://ratsgo.github.io/deep%20learning/2017/09/25/gradient/)**을 참고하세요.

Gradient Boosting은 간단히 말해 Gradient Descent를 이용해 Boosting을 하는 방법입니다.

복습해보자면, AdaBoost에서는 잘 맞추기 위해 이전 시점에서 잘 맞추지 못한 객체에 대해 더 잘 맞추는 모형을 지향했습니다.

GBM에서는 이전 시점에서 잘 맞추지 못한 **손실**에 대해 더 잘 맞추는 모형을 지향합니다.

즉 GBM은 Y에 관심이 있습니다. 아래의 그림을 보고 구체적으로 설명드리겠습니다.

![그림14](https://imgur.com/sfDpHMb.png)

Ensemble Learning은 크게 Bagging과 Boosting으로 나눌 수 있다고 했습니다.

Boosting은 다시 AdaBoost와 GBM으로 나눌 수 있습니다.

AdaBoost는 이전 시점의 결과에 의해 X를 조절 -잘 맞추지 못한 객체에 대한 추출 확률을 높임- 하나 GBM은 이전 시점의 결과에 의해 Y를 조절합니다.

Y를 조절한다는 것은 이전 시점에서 설명하지 못한 잔차를 다음 시점의 반응변수로 사용한다는 의미입니다.

아래의 그림을 보시면 GBM은 이전 모형의 잔차를 설명하기 위해 학습하는 것을 알 수 있습니다.

![그림15](https://imgur.com/YVtmR4g.png)

이때 Loss function이 L2인 경우에는

![그림16](https://imgur.com/fLmBqTv.png)

와 같이 Gradient가 계산됩니다. 따라서 잔차를 학습하는 방향은 Gradient의 반대방향임을 알 수 있습니다.

GBM은 L2말고도 다양한 Loss function의 적용이 가능하며

이는 뒤에서 **[Python 프로그램](https://www.python.org/)**으로 구현할 때 보여드리도록 하겠습니다.

예측 모형에 적용할 수 있는 Loss function의 예는

![그림17](https://imgur.com/9J0hgsy.png)

와 같고 분류 모형에 적용할 수 있는 Loss function의 예는

![그림18](https://imgur.com/o1cj7Ij.png)

와 같습니다.

GBM은 이전 시점의 잔차를 학습하기 위해 작동하기 때문에 Training set에 Overfitting될 위험이 굉장히 높습니다.

따라서 GBM은 몇 가지 방법을 사용하여 이를 방지합니다.

**첫 번째는 Subsampling입니다.**

여기서 말하는 Subsampling은 Bagging에서 사용되는 Bootstrap과는 다릅니다.

잔차를 학습하는 과정에서 Training set에 Overfitting되는 상황을 조금이나마 방지하고자 비복원 추출을 통해 Subsampling을 하는데

Bagging도 가능하긴 합니다.(즉 GBM에서의 Subsampling은 복원 / 비복원이 모두 가능하며 Bootstrap은 복원 추출만 가능합니다.)

**두 번째는 Shrinkage입니다.**

T개의 앙상블이 잔차를 학습하는 동안 계속해서 Training set의 잔차를 외우는 상황이 발생합니다.

이때 Shrinkage는 뒤로 갈수록(앙상블의 수가 T에 가까울수록) Shrinkage factor를 곱하여 그 영향을 줄입니다.

즉 학습률이 alpha일 때 shrinkage factor인 lambda에 의해

![그림19](https://imgur.com/3kr1GXN.png)

가 되면서 Shrinkage 효과를 얻습니다.

**세 번째는 학습 조기종료입니다.**

이는 Ensemble Learning이 Training set의 잔차를 모두 외우기 전에 미리 학습을 종료시키는 것입니다.

이때 조기 종료하는 Threshold는 분석자의 Hyperparameter입니다.

아래의 그림은 **[고려대학교 강필성 교수님의 수업자료](https://github.com/pilsung-kang/Business-Analytics/blob/master/04%20Ensemble%20Learning/04_Ensemble%20Learning.pdf)**에서 발췌 및 편집한 것으로, 조기 종료하면 성능이 올라갈 수 있는 포인트가 존재함을 알 수 있습니다.

![그림20](https://imgur.com/vtlIr0t.png)

**네 번째는 터미널 노드의 데이터 수입니다.**

지금부터 설명드릴 부분은 Tree-based GBM에 해당되는 방법입니다.

Tree-based GBM을 사용하는 경우, 단일 트리의 터미널 노드에 속하는 최소 데이터 수(Min.obj)를 제한하고

Min.obj보다 작은 수를 갖는 터미널 노드에 대한 분할을 무시합니다.

이렇게 되면 이상치와 같은 특이값 때문에 발생하는 변동폭을 줄일 수 있어서 성능 향상에 도움이 될 수 있습니다.

GBM에서의 마지막 내용입니다.

Tree-based GBM은 **[Random Forest](https://ratsgo.github.io/machine%20learning/2017/03/17/treeensemble/)**와 같은 다른 Tree-based Ensemble들과 같이 변수의 중요도를 파악할 수 있습니다.

변수의 중요도는 모형을 예측 또는 분류하는 동안 중요하다고 판단되는 변수에 대해 높은 값이 나오도록 설계되어 있습니다.

주의할 점은 로지스틱 회귀분석과 같이 변수에 대한 통계적 유의성을 보장하는 것은 아니라는 것입니다.

즉 변수들의 **상대적인 중요도**를 보여주는 것입니다.

Tree-based GBM에서 변수의 중요도는 Information Gain(IG)을 이용해 계산됩니다.

단일 의사결정 나무모형 T에서 j번째 변수의 중요도는

![그림21](https://imgur.com/X5n24Q6.png)

로 계산되며 Gradient Boosting에서는

![그림22](https://imgur.com/QD9TFOw.png)

로 계산됩니다.

이를 단순화된 예로 설명드리면, 첫 번째 단일 의사결정 나무모형이

![그림23](https://imgur.com/45GppL0.png)

와 같고 두 번째 단일 의사결정 나무모형이

![그림24](https://imgur.com/SSyzyye.png)

와 같으면 각 변수의 IG는

![그림25](https://imgur.com/DEhYtXV.png)

로 정리할 수 있기 때문에 가장 중요한 변수는 X2라고 할 수 있습니다.

<br>

<h3>GBM in Python</h3>

지금까지 설명한 GBM을 **[Python 프로그램](https://www.python.org/)**을 통해 구현해보겠습니다.

코드는 **[이곳](https://github.com/pilsung-kang/Business-Analytics/blob/master/04%20Ensemble%20Learning/Tutorial%2012%20-%20AdaBoost%20and%20GBM/BA_gbm.ipynb)**을 참고하였고 추가로 Subsampling, Shrinkage, Early stop을 구현하였습니다.

<h4>Source code</h4>

**[Python 프로그램](https://www.python.org/)**을 통해 구현한 Tree-based GBM입니다.

이 코드는 Tree-based GBM을 하나의 Python class로 구축하였습니다.

지정해주어야 하는 변수로는

예측변수(x), 반응변수(y),
x 분기 기준(x_split_val), 앙상블 크기(tree),
조기 종료 Threshold(error), Subsampling 크기(subsamp_ratio),
Loss function(L1='Absolute', L2='Deviance', Huber loss='Huber')

들이 존재하며 x와 y를 제외하고 default 값은 지정되어 있습니다.

```python
#Ver.3: Early stop + Subsampling + Loss function
from sklearn.metrics import mean_squared_error
class GradientBoosting_Machine():
	"""
	Initialization
	"""
    def __init__(self, x, y, x_split_val=0.1, tree=100, error=0.1, loss_func='Deviance', subsamp_ratio=1.0, shrinkage_factor=1.0, delta=0.01):
        self.x = x
        self.y = y
        self.x_split_val = x_split_val
        self.tree = tree
        self.error = error
        self.loss_func = loss_func
        self.subsamp_ratio = subsamp_ratio
        self.shrinkage_factor = shrinkage_factor
        self.delta = delta # only use when apply Huber loss function: loss_func='Huber'
    """
    Subsamping
    """
    def subsampling(self, x, y):
        data_len = shape(self.x)[0]
        my_size = round(self.subsamp_ratio, 1) * 10
        wor = random.choice(arange(0, data_len), size=my_size, replace=False)
        self.x = x[wor]
        self.y = y[wor]
    """
    Split function
    """
    def split(self, x, y, x_split_val):
        x_min = sort(x)[0]
        x_max = sort(x)[len(x)-1]
        x_thresholds = arange(x_min, x_max, x_split_val)
        y_list, left_list, right_list = [], [], []
        for i in range(len(x_thresholds)):
            indices_left = [k for k, x_k in enumerate(x) if x_k <= x_thresholds[i]]
            indices_right = [k for k, x_k in enumerate(x) if x_k > x_thresholds[i]]
            x_left = array([x[k] for k in indices_left])
            y_left = array([y[k] for k in indices_left])
            x_right = array([x[k] for k in indices_right])
            y_right = array([y[k] for k in indices_right])
            split_y_left = mean(y_left)
            y_left_residual = y_left - split_y_left
            left_list.append(split_y_left)
            split_y_right = mean(y_right)
            y_right_residual = y_right - split_y_right
            right_list.append(split_y_right)
            y_list.append(append(y_left_residual, y_right_residual))
        return y_list, x_thresholds, left_list, right_list
    """
    Loss Function
    """
    def loss(self, selected_list):
        # Loss function 추가
        Dev = sum(0.5*((selected_list)**2))
        Abs = sum(abs(selected_list))
        Hub = sum( (self.delta**2) * (sqrt(1+(selected_list/self.delta)**2) -1 ))
        # 추가하고 싶은 Loss function 정의
        set_loss = [Dev, Abs, Hub]
        # set_loss = [Dev, Abs, 추가하고 싶은 Loss function 삽입]
        set_lossname = ['Deviance', 'Absolute', 'Huber']
        #set_lossname = ['Deviance', 'Absolute', 추가하고 싶은 Loss function 호출명 삽입]
        for i in range(len(set_loss)):
            if set_lossname[i] == self.loss_func:
                loss_fun = set_loss[i]
                return loss_fun   
            else:
                pass
    """
    Gradient Boosting
    """
    def select_residual(self):
        new_x = None
        new_y = self.y
        l_ = None
        r_ = None
        min_error = inf
        new_x_list, new_y_list, split_y_left_list, split_y_right_list = [], [], [], []
        beststump = {}
        for s in range(self.tree):
            selected_y_list, x_thresholds, left_list, right_list = self.split(self.x, new_y, self.x_split_val)
            new_x_list.append(new_x)
            new_y_list.append(new_y)
            split_y_left_list.append(l_)
            split_y_right_list.append(r_)
            q_list = []
            for u in range(len(selected_y_list)):
                q = selected_y_list[u]
                """
                Shrinkage
                """
                q_list.append(self.loss(q) * self.shrinkage_factor)
                min_error = min(q_list)
                new_y = selected_y_list[q_list.index(min(q_list))]
                new_x = x_thresholds[q_list.index(min(q_list))]
                l_ = left_list[q_list.index(min(q_list))]
                r_ = right_list[q_list.index(min(q_list))]
            """
            Early Stopping
            """
            if (min_error < self.error):
                beststump['s'] = s
                break
            else:
                continue
        return new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, min_error
    """
    Predicting function
    """
    def Predict(self, testdata):
        new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, min_error = self.select_residual()
        for i in range(len(new_x_list)):
            if new_x_list[i] == None:
                new_x_list[i] = self.x[i]
            else:
                pass
        predicted_val_list = []
        for m in range(len(testdata)):
            residual_sum = []
            for n in range(len(split_y_left_list)-1):
                if testdata[m] <= new_x_list[n+1]:
                    residual_sum.append(split_y_left_list[n+1])
                else:
                    residual_sum.append(split_y_right_list[n+1])
            print(testdata[m], "predict value :", sum(residual_sum))
            predicted_val_list.append(sum(residual_sum))
        return predicted_val_list
```
Tree-based GBM을 **[예시 데이터](https://github.com/pilsung-kang/Business-Analytics/blob/master/04%20Ensemble%20Learning/Tutorial%2012%20-%20AdaBoost%20and%20GBM/BA_gbm(ver1).py)**에 적용해보겠습니다.

```python
from numpy import *
def loadSimpleData():
	train_data = array([[1], [1.8], [3], [3.7], [4], [4.2], [4.5], [5], [5.3], [5.4], [5.9], [6.8], [7], [7.5], [7.6], [7.7],
	                 [8.1], [8.3], [9], [9.5]])
	train_label = array([2.2, 0.7, 0.6, 0.9, 1, 1.4, 1.5, 0.8, -0.7, -0.8, -0.9, 0.4, 0.6, -0.7, -1.0, -1.2, -1.5, 1.6, -1.1, 0.9])
	# train_data = np.array([[1], [1.8], [3], [3.7], [4], [4.2], [4.5], [5], [5.3], [5.4]])
	# train_label = np.array([2.2, 0.7, 0.6, 0.9, 1, 1.4, 1.5, 0.8, -0.7, -0.8])
	test_data = array([[1], [3], [3.7], [4.2], [4.5]])
	test_label = [2.2, 0.6, 1, 1.4, 1.5]
	return train_data, train_label, test_data, test_label

train_data, train_label, test_data, test_label = loadSimpleData()
# Fit: Tree-based GBM
GBM = GradientBoosting_Machine(x=train_data, y=train_label, x_split_val=0.1, tree=160, error=0.1, loss_func='Deviance')
```
이제 Test set에 적용해보겠습니다.
```python
new_x_list, new_y_list, split_y_left_list, split_y_right_list, beststump, min_error = GBM.select_residual()
print("best_tree:", beststump.get('s'))
predicted_label = GBM.predict(testdata=test_data)
print("test_label: ", test_label)
print("predicted_label: ", predicted_label)
mse = mean_squared_error(test_label, predicted_label)
print("MSE: %.4f" % mse)
```
```python
[Out]:
best_tree: 50
[1.] predict value : 2.151100030398735
[3.] predict value : 0.689337453853373
[3.7] predict value : 0.9666402446390674
[4.2] predict value : 1.3834889168561655
[4.5] predict value : 1.3834889168561655
test_label:  [2.2, 0.6, 1, 1.4, 1.5]
predicted_label:  [2.151100030398735, 0.689337453853373, 0.9666402446390674, 1.3834889168561655, 1.3834889168561655]
MSE: 0.0051
```

적용 결과를 보시면 Test MSE는 0.0051로 우수한 성능을 보이고 있습니다.

<h4>Scikit learn code</h4>

**[Python 프로그램](https://www.python.org/)**에서 모델링을 위한 **[scikit learn 라이브러리](https://scikit-learn.org/stable/index.html)**를 사용하여 같은 데이터를 분석해보겠습니다.

scikit learn에서는 GBM 기반의 Classifier를 **[sklearn.ensemble.GradientBoostingClassifier()](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)**로 제공하고 있습니다.

예측 모형인 경우는 **[sklearn.ensemble.GradientBoostingRegressor()](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)**를 사용하시면 됩니다.

scikit learn을 통해 cancer 데이터를 분석해보겠습니다.

```python
# Loading data set: Cancer data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Dim: {}".format(cancer.data.shape))
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(cancer.data, cancer.target, random_state=40)
print("Train_Dim: {}".format(train_x.shape))
print("Test_Dim: {}".format(test_x.shape))
```
```python
[Out]:
Dim: (569, 30)
Train_Dim: (426, 30)
Test_Dim: (143, 30)
```
```python
from sklearn.ensemble import GradientBoostingClassifier
GB_Classifier = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, max_depth=3,
                                 # 'subsample' is that be used for fitting the individual base learners
                                 # If smaller than 1.0 this results in Stochastic Gradient Boosting.
                                 # So, choosing subsample < 1.0 leads to a reduction of variance and increase in bias.
                                 subsample=1.0,)
# Training
gbc = GB_Classifier.fit(train_x, train_y)
print("Training acc: {}".format(gbc.score(train_x, train_y))) # Training acc == 100 => Overfitting
# Predicting
print("Testing acc: {:.4f}".format(gbc.score(test_x, test_y)))
```
```python
[Out]:
Training acc: 1.0
Testing acc: 0.9650
```
분석 결과 Training set에 Overfitting된 모형입니다.

따라서 Tree의 깊이를 줄여보도록 하겠습니다.

```python
# Prevent Overfitting
# 1st => reducting the depth of tree
Max_Depth_of_tree = 1 # Always, should be less than 5
GB_Classifier_prun = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, max_depth=Max_Depth_of_tree, subsample=1.0)
gbc_prun = GB_Classifier_prun.fit(train_x, train_y)
print("Training acc: {}".format(gbc_prun.score(train_x, train_y))) # Training acc == 100 => Overfitting
print("Testing acc: {:.4f}".format(gbc_prun.score(test_x, test_y)))
```
```python
[Out]:
Training acc: 0.9953
Testing acc: 0.9650
```

그래도 Training set을 너무 외우는 경향이 있기 때문에 Subsampling을 해보면
```python
# 3rd => reducting the sampling size (& learning rate)
Sub_sampling = 0.3
GB_Classifier_subs = GradientBoostingClassifier(loss='deviance', learning_rate=Learning_rate, n_estimators=100, max_depth=3, subsample=Sub_sampling)
gbc_subs = GB_Classifier_subs.fit(train_x, train_y)
print("Training acc: {:.4f}".format(gbc_subs.score(train_x, train_y))) # Training acc == 100 => Overfitting
print("Testing acc: {:.4f}".format(gbc_subs.score(test_x, test_y)))
```
```python
[Out:]
Training acc: 0.9812
Testing acc: 0.9720
```
Training Accuracy는 떨어지고 Test Accuracy는 더욱 높아진 것을 알 수 있습니다.

아래의 그림은 분석 결과를 비교한 것입니다.

![그림26](https://imgur.com/Co88Dj8.png)

Subsampling을 하는 경우 Training set에 Overfitting하는 경향이 줄어드는 것을 알 수 있습니다!

지금까지 Ensemble Learning 중 Boosting, 그 중 대표적인 Boosting 방법인 AdaBoost와 GBM에 대해 알아보았습니다.
