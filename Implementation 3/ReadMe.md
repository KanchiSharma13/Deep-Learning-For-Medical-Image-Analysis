Implementing: [Deep residual transfer learning for automatic diagnosis and grading of diabetic retinopathy](https://www.sciencedirect.com/science/article/pii/S0925231220316520)<br><br>
In previous implementations involving IDRiD, data imbalance and overfitting are major problems associated with low test accuracy. This paper introduces comprehensive methods to address these issues. Although the original paper was implemented on the MESSIDOR dataset, our study uses the IDRiD dataset and achieves <b>superior results</b>. Let's break down the steps: <br>
<h3>Pre-Processing</h3><br>
1) Centre cropping images to 900*900<br>
2) Scaling image intensity to [0,1] <br>
3) Random horizontal flipping(p=0.5) <br>
<br><br>
<h3>Research Methodology</h3><br>
The paper uses transfer learning approach on a ResNet-based system(ResNet18 & ResNet50). The layers are unfrozen as per the two training stages proposed(in the next section). All parameters on the highlighted layers are retrained in the finetuning step. Note that in ResNet-50, two re-training cases are proposed: in ResNet-50a only the last fully-connected layers are re-trained, but for ResNet-50b (and ResNet-18) the last residual layer has also been re-trained with the new retina data. The model is assigned to classify 5 classes: Grade 0(Normal), Grade 1, Grade 2, Grade 3, and Grade 4.<br>
![image](https://github.com/KanchiSharma13/Internship-2024-code-files/assets/98634679/ce3358a5-9088-4a61-a378-7118af5c415a)

<br><br>
<h3>Training</h3><br>
The training process using the TL paradigm is performed in two stages. First, the layers whose weights will be reused, are fixed to avoid their weights updating via backpropagation. Then, Adadelta (lr = 1.0, p=0.9) is used to update the parameters of the selected layers, and early stopping is used to set the weights to the minimum loss iteration. Early stopping is described here as the iteration with minimum loss over a 20-epoch training. Finally, in a further finetuning stage, we use Stochastic Gradient Descent (SGD) with a low learning rate (lr = 0.001) over 50 epochs and early stopping for the final trained model.

