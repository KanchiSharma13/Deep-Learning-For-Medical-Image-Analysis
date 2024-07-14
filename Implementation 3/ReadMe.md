<h2> Implementing: <a href="https://www.sciencedirect.com/science/article/pii/S0925231220316520">Deep residual transfer learning for automatic diagnosis and grading of diabetic retinopathy</a>(using PyTorch)</h2><br><br>
In previous implementations involving IDRiD, data imbalance and overfitting are major problems associated with low test accuracy. This paper introduces comprehensive methods to address these issues. Although the original paper was implemented on the MESSIDOR dataset, our study uses the IDRiD dataset and achieves <b>superior results</b>. Let's break down the steps: <br>
<h3>Pre-Processing</h3>
1) Centre cropping images to 900*900<br>
2) Scaling image intensity to [0,1] <br>
3) Random horizontal flipping(p=0.5) <br>
<br><br>

<h3>Research Methodology</h3>
The paper uses transfer learning approach on a ResNet-based system(ResNet18 & ResNet50). The layers are unfrozen as per the two training stages proposed(in the next section). All parameters on the highlighted layers are retrained in the finetuning step. Note that in ResNet-50, two re-training cases are proposed: in ResNet-50a only the last fully-connected layers are re-trained, but for ResNet-50b (and ResNet-18) the last residual layer has also been re-trained with the new retina data. The model is assigned to classify 5 classes: Grade 0(Normal), Grade 1, Grade 2, Grade 3, and Grade 4.<br><br>
<center><img src="https://github.com/KanchiSharma13/Internship-2024-code-files/blob/main/Implementation%203/Images/ResNet.png" align="center"></center><br>
<center><img src="https://github.com/KanchiSharma13/Internship-2024-code-files/blob/main/Implementation%203/Images/Vgg%20alexnet.png" align="center"></center><br>
<center><img src="https://github.com/KanchiSharma13/Internship-2024-code-files/blob/main/Implementation%203/Images/Layers%20info.png" align="center"></center>
<h5 align="center">Figs: Schema of the architectures for the different network topologies and the retrained layers</h5> 

<br><br>
<h3>Training</h3>
The training process using the TL paradigm is performed in two stages. First, the layers whose weights will be reused, are fixed to avoid their weights updating via backpropagation. Then, Adadelta (lr = 1.0, p=0.9) is used to update the parameters of the selected layers, and early stopping is used to set the weights to the minimum loss iteration. Early stopping is described here as the iteration with minimum loss over a 20-epoch training. Finally, in a further finetuning stage, we use Stochastic Gradient Descent (SGD) with a low learning rate (lr = 0.001) over 50 epochs and early stopping for the final trained model.
<br><br>

<h3>Evaluation</h3>
To compare the ResNet-based system with the state of the art, we have trained and tested under the same conditions three additional, well-known, deep learning models: AlexNet, VGGnet-16, and VGGnet-19, all tested within a 5-fold stratified cross-validation.
<br><br>

<h3>Results</h3>
The results are provided for three different setups. In the first approach, we present the most challenging scenario: a full multiclass grading using the different models on the five different labels(classes: 0, 1, 2, 3, 4). Then, we test an aggregation of the 1 and 2 labels(classes: 0, 1+2, 3, 4). Finally, we also provide an analysis of the performance of a binary classification scenario covering grades 0 and 4. Evaluation parameters such as sensitivity and specificity (Sens. and Spec., only for binary classification) and the Receiver Operating Characteristic (ROC) curve, as well as the area under the ROC (AUC) were computed per class.<br>
<h5>Case 1: Multiclass Classification(Grade 0, 1, 2, 3, 4)</h5>
<img src="https://github.com/KanchiSharma13/Internship-2024-code-files/blob/main/Implementation%203/Images/case1.png" width="600" height="350"><br>
<h5>Case 2: 4 class scenario(Grade 0, 1+2, 3, 4)</h5>
<img src="https://github.com/KanchiSharma13/Internship-2024-code-files/blob/main/Implementation%203/Images/case2.png" width="600" height="240"><br>
<h5>Case 3: Binary Classification(Grade 0 & 3)</h5>
<img src="https://github.com/KanchiSharma13/Internship-2024-code-files/blob/main/Implementation%203/Images/case3.png" width="600" height="210"><br>

<h3>Conclusion & Discussion</h3>
Experiments performed have shown the superiority of convolutional networks including residual blocks with respect to non-residual architectures such as VGGnet & AlexNet. Having a huge number of parameters makes it difficult to learn simpler functions. This way, residual blocks, which add up the result of the previous layer to the output of the former bypassing it, ease the learning of simple functions. On the other hand, they also mitigate the vanishing gradient problem that hinders the training process. The experiments we carried out show that our work has achieved better results in cases 1 & 2 than in the original study. 

