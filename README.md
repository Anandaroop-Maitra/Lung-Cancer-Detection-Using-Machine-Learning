Hi, in this repository, I have worked on Lung Cancer Detection Using Machine Learning, by studying the performance of CNN and Transfer Learning Algorithm.


Lung cancer remains one of the leading causes of cancer-related mortality worldwide, necessitating the development of advanced diagnostic tools to improve early detection and treatment outcomes. This project explores the application of machine learning techniques, specifically Convolutional Neural Networks (CNN) and Transfer Learning algorithms, in the detection of lung cancer. The primary objective is to evaluate the performance and efficacy of these methods in identifying malignant lung nodules from medical imaging data.


I have constructed a comprehensive dataset of lung images, incorporating both healthy and cancerous samples, sourced from publicly available medical imaging repositories. The dataset was pre-processed to enhance image quality and augment the data through techniques such as rotation, scaling, and flipping to improve model robustness.


![Screenshot 2023-11-29 123329](https://github.com/Anandaroop-Maitra/Lung-Cancer-Detection-Using-Machine-Learning/assets/62735860/f9cd9eb2-6e34-42da-96de-b058f2bfaf6c)






A CNN model was designed and trained from scratch to classify the images, leveraging its powerful feature extraction capabilities. Simultaneously, a Transfer Learning approach was implemented using pre-trained models like VGG16, ResNet50, and InceptionV3, which were fine-tuned on our specific dataset. This approach leverages the knowledge from large-scale image classification tasks to improve performance on our lung cancer detection task.

Project Methodology:


![Screenshot 2024-05-14 201658](https://github.com/Anandaroop-Maitra/Lung-Cancer-Detection-Using-Machine-Learning/assets/62735860/d415cad4-0bfe-4960-8521-48dab7a57f13)



The performance of the models was evaluated using metrics such as accuracy, precision, recall, and the F1 score. Our findings indicate that the Transfer Learning models, particularly ResNet50, outperformed the custom-built CNN in terms of accuracy and computational efficiency. The study highlights the potential of Transfer Learning to enhance the diagnostic accuracy of lung cancer detection systems significantly.

CNN Model Performance:


![Screenshot 2024-04-30 143542](https://github.com/Anandaroop-Maitra/Lung-Cancer-Detection-Using-Machine-Learning/assets/62735860/2d89a061-444c-4241-9327-7bdb563b8021)


Transfer Learning Model Performance:


![Screenshot 2024-04-30 151301](https://github.com/Anandaroop-Maitra/Lung-Cancer-Detection-Using-Machine-Learning/assets/62735860/01f961fe-4ec3-45b3-9b07-fcff89baa286)





This research demonstrates that advanced machine learning models, particularly those utilizing Transfer Learning, can be instrumental in the early detection of lung cancer. Future work will focus on integrating these models into clinical workflows and exploring real-time application capabilities to assist radiologists in making faster and more accurate diagnoses.


The project dataset can be downloaded from the link: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images


The project is entirely compiled and run in Jupyter Notebook.




