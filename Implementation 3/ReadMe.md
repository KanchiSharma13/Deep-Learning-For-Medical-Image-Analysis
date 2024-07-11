<h1>Implementing: [Deep residual transfer learning for automatic diagnosis and grading of diabetic retinopathy](https://www.sciencedirect.com/science/article/pii/S0925231220316520)<br><br></h1>
In previous implementations involving IDRiD, data imbalance and overfitting are major problems associated with low test accuracy. This paper introduces comprehensive methods to address these issues. Although the original paper was implemented on the Messidor dataset, our study uses the IDRiD dataset and achieves superior results. Let's break down the steps: <br>
<h3>Pre-Processing</h3><br>
1) Centre cropping images to 900*900<br>
2) Scaling image intensity to [0,1] <br>
3) Random horizontal flipping(p=0.5) <br>

