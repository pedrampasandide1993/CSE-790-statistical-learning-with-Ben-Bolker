<div style="text-align:center;font-size:20px">
  CSE 790 - Assignment 1
</div>
<br>
<div style="text-align:center;font-size:15px">
  Pedram Pasandide - 400417813
</div>

<br>
<br>
<br>
<br>

**<span style="font-size:16px">Question 1 </span>**

The author is suggesting to reduce the number features. There are some methods reduce the number of observations. Any sensitivity analysis investigating the effects of features on the output can be helpful. The ones with less effects can be eliminated to reduce the complexity of the model or even avoiding the over-fitting. 

Another way is to see if two or more independent variables are highly correlated with one another. To do so, it is necessary to investigate multicollinearity. Multicollinearity among independent variables will result in less reliable statistical inferences. Variance Inflation Factor (VIF) can evaluate the multicollinearity. If $VIF>10$ then multicollinearity is high. VIF is defined by the following formula.

$$VIP = \frac{1}{1 − R^{2}}$$
where $R^{2}$ is R-squared.

Introducing a new variable which is a mathematical combination of two or more variables can be also used to avoid Simpson paradox. In all mentioned methods, the physics of the problem and its limitations should be considered.
<br>
<br>
**<span style="font-size:16px">Question 2 </span>**

Not Solved.
<br>
<br>
**<span style="font-size:16px">Question 3 </span>**

The same as MSE for minimization, we can split the integration into two, one from -∞ to m, and the other one from m to +∞. Then we can differentiate the both sides of the equation to find m. Eventually, we have:

$$\frac{dE|Y-m|}{dm}=\frac{dI_{1}}{dm}+\frac{dI_{2}}{dm}=0$$

where

$$I_1 = \int_{-∞}^{m} p(y)*|Y-m|dy$$
$$I_2 = \int_{m}^{+∞} p(y)*|Y-m|dy$$

$$\frac{dI_1}{dm} = 0 +\int_{-∞}^{m} \frac{\partial}{\partial m}p(y)*|Y-m|dy = \int_{-∞}^{m} p(y)dy$$

$$\frac{dI_2}{dm} = 0 -\int_{m}^{+∞} \frac{\partial}{\partial m}p(y)*|Y-m|dy =- \int_{m}^{+∞} p(y)dy$$

Showing that $\frac{dI_1}{dm} = - \frac{dI_2}{dm}$ and based on the first equation:

$$P(X≥m) = P(X≤m) = 0.5$$

meaning that the median minimizes the error. Since MSE is more sensitive to changes, MSE would be a better choice.
<br>
<br>
**<span style="font-size:16px">Question 4 </span>**

Since a linear smoother is a sample me the definition can written as:

$$\sum y_i \hat{w}(x_i,x) = [\sum y_i]/n$$

meaning $\hat{w}(x_i,x) = 1/n$. And of a squared matrix with size of n $tr(w) = n*(1/n) = 1$ which is the degree of the freedom of linear smoother.
<br>
<br>
**<span style="font-size:16px">Question 5 </span>**

Not Solved.
<br>
<br>
**<span style="font-size:16px">Question 6 </span>**

The data was imported using `read_table()` in `pandas`. `loc` and `iloc` functions were used to only keep 2's and 3's.
I don't understand why the question is asking to use linear regression to do the classification since linear regression 
is used only for continuous values. A threshold, for example equal to (2+3)/2, can be defined to classify the output.
However, in this homework, I have used logistic regression, with max iteration of 200, and KNN in scikit-learn. Following
formulation has been used to evaluate the error.  

$$Error = \frac{y_{real}-y_{prediction}}{N}$$

where N is the number of samples.

The results are shown in the following table. 

| Methods | Train error | Test error | Train score |
|:-------:|:-----------:|:----------:|:-----------:|
| Logistic Regression  |   0    |   0   |   1    |
| KNN for k = 1  |   0   |   -0.00824   |   1    |
| KNN for k = 3  |   0.00072    |   -0.00824   |   0.99496    |
| KNN for k = 5  |   -0.00143    |   -0.00143   |   0.99424    |
| KNN for k = 7  |   -0.00215    |   -0.01648   |   0.99352    |
| KNN for k = 15  |   -0.00503    |   -0.02197   |   0.99064    |

For $k = 1$ the accuracy is high which might be because of over fitting, as well as logistic regression. The code can be found at `question6_Pedram.py`.

