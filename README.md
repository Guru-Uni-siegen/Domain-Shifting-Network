# Domain-Shifting-Network

In this repository we provide python code to reproduce results from the paper:

> **A Simple Domain Shifting Network for Generating Low Quality Images**

> ICPR 2020 (http://arxiv.org/abs/2006.16621)

> Guruprasad Hegde, Avinash Nittur Ramesh, Kanchana Vaishnavi Gandikota, Roman Obermaisser, Michael Moeller

Deep Learning systems have proven to be extremely successful for image recognition tasks for which significant amounts of training data is available, e.g., on the famous ImageNet dataset. We demonstrate that for robotics applications with cheap camera equipment, the low image quality, however, influences the classification accuracy, and freely available data bases cannot be exploited in a straight forward way to train classifiers to be used on a robot. As a solution we propose to train a network on degrading the quality images in order to mimic specific low quality imaging systems. Numerical experiments demonstrate that classification networks trained by using images produced by our quality degrading network along with the high quality images outperform classification networks trained only on high quality data when used on a real robot system, while being significantly easier to use than competing zero-shot domain adaptation techniques.

## Performance comparison for 5-way classification. 


|Approach					|	*Standard  |	   *Cozmo	|
|---------------------------|-------------|-------------|
|Source Supervised			|	92.87	  |		73.49	|
|Ours Unsupervised			|	91.66	  |		77.56	|
|Ours zero-shot				|	92.09	  |		76.39	|
|Cozmo Supervised			| 84.88	  |		80.15	|
###### *The reported numbers are classification accuracy

Results in the paper can be repoduced using the code avaiable in this repository and data used for the experiments are available in the below google drive link

https://drive.google.com/file/d/1t9phYorkTLhRb86vRMZ_-3XdpDFdCkla/view?usp=sharing

# Install

Pip requirement file can be installed using the below command

```
pip install -r requirements.txt
