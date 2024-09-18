# DenseNet and Image Classification

<p align="justify">
<strong>DenseNet (Densely Connected Convolutional Networks)</strong> is a deep learning architecture designed to address the vanishing gradient problem and improve feature reuse in convolutional neural networks. Introduced by Gao Huang and his team in 2017, DenseNet enhances traditional convolutional networks by connecting each layer to every other layer in a feed-forward fashion. This means that each layer receives inputs from all preceding layers and passes its own feature maps to all subsequent layers, creating a dense connectivity pattern.
</p>
<p align="justify">
This architecture leads to improved gradient flow, more efficient feature reuse, and a reduction in the number of parameters compared to traditional networks. DenseNet models have achieved state-of-the-art performance on various benchmark datasets and have become popular for tasks such as image classification and object detection.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_DenseNet_ImageClassification_mexgtkug.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_densenet_train_sgxbapee.py --data_dir ../dataset --num_epochs 10 --batch_size 16 --model_path aicandy_model_out_ddmalncc/aicandy_model_pth_silsegko.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_densenet_test_beyugdam.py --image_path ../aicandy_true_dog.jpg --model_path aicandy_model_out_ddmalncc/aicandy_model_pth_silsegko.pth
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_convert_to_onnx_dvicqkqc.py --model_path aicandy_model_out_ddmalncc/aicandy_model_pth_silsegko.pth --output_path aicandy_model_out_ddmalncc/aicandy_model_onnx_ptspsbyh.onnx
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-densenet-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




