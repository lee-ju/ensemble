이번 포스팅에서는 앙상블 기법 중 Boosting에 해당하는 AdaBoost와 Gradient Boosting Machine에 대해 살펴보겠습니다.

해당 포스팅은 **[고려대학교 산업경영공학과 강필성 교수님](https://github.com/pilsung-kang)** 강의를 참고했음을 밝힙니다.

<h1>Ensemble Learning</h1>
<h3>앙상블은 무엇일까요?</h3>
다양한 데이터 분석 기법들은 대부분 단일모형을 가지고 예측 또는 분류를 진행합니다.

**[Ensemble Learning](https://en.wikipedia.org/wiki/Ensemble_learning)**은 다양한 단일모형의 조합을 통하여 성능을 끌어 올리기 위하여 고안된 방법입니다.

앙상블을 구성하는 단일모형들은 조합 안에서 결과**(예측값 또는 예측 클래스)**를 피력하고, 최종결과는 단일모형들의 표결에 따라 결정됩니다.

데이터 분석에 이용되는 모형들은 대부분 아래와 같이 표현이 가능합니다.

![fig-adaboost1](https://user-images.githubusercontent.com/38771583/122848607-de8dbd80-d344-11eb-8e88-74719c7f8382.png)

이때 각 객체마다의 MSE는

![fig-adaboost2](https://user-images.githubusercontent.com/38771583/122849799-27df0c80-d347-11eb-8482-4dc58e8373d7.png)

와 같이 분해가 가능합니다.

따라서 우리는 통제가 가능한 오차인 편향과 분산에 대해 관심을 가질 필요가 있습니다.

편향은 실제값과 추정된 값들의 차이로 설명할 수 있습니다. 그리고 분산은 객체들의 퍼져있는 정도로 설명할 수 있습니다.

이러한 의미로 보았을 때, 아래의 그림에서 왼쪽은 낮은 편향과 높은 분산을 가지는 경우에 해당합니다.

반대로 아래의 그림에서 오른쪽은 높은 편향과 낮은 분산을 가지는 경우입니다.
<h2></h2>
![fig-adaboost3](https://user-images.githubusercontent.com/38771583/122849827-33323800-d347-11eb-850a-b942c446bf99.png)
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

![fig-adaboost4](https://user-images.githubusercontent.com/38771583/122849837-3a594600-d347-11eb-8a0a-2842c0135841.png)

와 같고 오차는 평균이 0이고 서로 독립이라는 아래와 같은
![fig-adaboost5](https://user-images.githubusercontent.com/38771583/122849843-3e856380-d347-11eb-941a-0b33ad038f3a.png)

가정을 하게 됩니다. 그러면 **M**개의 각 모형들에 의한 에러의 평균은

![fig-adaboost6](https://user-images.githubusercontent.com/38771583/122849857-447b4480-d347-11eb-9800-db24544a1a5c.png)

와 같고 **M**개의 개별 모형들의 에러들은

![fig-adaboost7](https://user-images.githubusercontent.com/38771583/122849864-480ecb80-d347-11eb-9f13-9f17d7202979.png)

로 계산됩니다. 이제 위에서 가정한 식에 의해

![fig-adaboost8](https://user-images.githubusercontent.com/38771583/122849874-4ba25280-d347-11eb-81be-b56c91e98866.png)

가 성립하나 이는 상당히 **이상적**인 가정에 의한 결과입니다.

따라서 이 결과는 코시 부등식에 의해

![fig-adaboost9](https://user-images.githubusercontent.com/38771583/122849896-50670680-d347-11eb-9c9f-2afb11389126.png)

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

![fig-adaboost10](https://user-images.githubusercontent.com/38771583/122849903-5361f700-d347-11eb-9f73-45ada5e25646.png)

즉 Bagging은 병렬처리가 가능하며 알고리즘에 독립적인 샘플을 사용하며, Boosting은 Sequential한 방법에 의해 작동합니다.

Bagging은 개별 모형의 분산을 낮추기 위해 사용되기 때문에, **분산이 큰 모형-유연/복잡한 모형-에 적용하는 것이 효과적**입니다.

Boosting은 개별 모형의 편향을 낮추기 위해 사용되기 때문에, **편향이 큰 모형-선형/단순한 모형-에 적용하는 것이 효과적**입니다.

<br>


<h3>Boosting: AdaBoost</h3>

Boosting 중 가장 유명한 AdaBoost는 이전 시점에서 잘 틀리는 부분을 다음 시점에서 더 잘 맞추도록 객체가 선택될 확률을 업데이트합니다.

AdaBoost는 weak 모형들을 주로 사용합니다.

여기서 말하는 weak 모형은 선형에 가까운 모형을 의미합니다.

AdaBoost는 아래의 알고리즘에 의해 작동합니다.

![fig-adaboost-11](https://user-images.githubusercontent.com/38771583/122849929-607ee600-d347-11eb-92bb-7b3d39c53356.png)

AdaBoost는 각 모형의 오분류율을 계산한 뒤, 오분류율을 이용해 alpha 함수를 만듭니다.

**alpha 함수는 크게 두 가지 역할을 합니다.** 첫 번째는 개별 모형에 대한 가중치이며 두 번째는 객체 선택 확률의 가중치입니다.

이에 대해 구체적으로 알아보겠습니다.

우선 오분류율이 0.5보다 큰 경우는 제외하기 때문에 오분류율의 범위는 [0, 0.5)가 됩니다.

오분류율이 0에 가까워지면 alpha 함수 값은 무한대로 커지게 되며 0.5에 가까워지면 alpha 함수 값은 0에 가까워지게 됩니다.

즉 alpha 함수는 잘 맞추는 모형(== 오분류율이 0에 가까운 모형)에 대해서는 큰 가중치를 주며

잘 맞추지 못하는 모형(== 오분류율이 0.5에 가까운 모형)에 대해서는 0에 가까운 가중치를 주게 됩니다.

그리고 현재 시점 t에서 모형이 정분류인 경우

![fig-adaboost-12](https://user-images.githubusercontent.com/38771583/122849954-696fb780-d347-11eb-85ab-a0a07cf67b6c.png)

이므로 해당 객체는 다음 시점에서 선택될 확률이 작아지게 됩니다.

반대로 현재 시점 t에서 모형이 오분류인 경우

![fig-adaboost-13](https://user-images.githubusercontent.com/38771583/122849961-6bd21180-d347-11eb-9f38-9915d2cd4fda.png)

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
