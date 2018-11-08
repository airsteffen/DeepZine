![Alt text](./resources/icon.png?raw=true "DeepZine")
_pronounced zeen_

# DeepZine

The thing that made [this Youtube video](https://www.youtube.com/watch?v=zH6guyxr0LI).

## Table of Contents
- [About](#about)
- [Requirements](#requirements) 
- [Documentation](#documentation)
- [PAQ (Probably Asked Questions)](#paq)

## About

This is a repository for a particular implementation of the Progressively Growing Generative Adversarial Network (PGGAN). This architecture was first developed by [Karras et al.](https://github.com/tkarras/progressive_growing_of_gans) in "Progressive Growing of GANs for Improved Quality, Stability, and Variation". The code that this repository was based on was developed by the Github user zhangqianhui's [Tensorflow implementation](https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow) of the PGGAN, although some significant changes have been made since.

While the classic implementations of the PGGAN so far have been to make highly-realistic [faces, objects](https://www.youtube.com/watch?v=XOxxPcy5Gr4), and [https://twitter.com/goodfellow_ian/status/937406530743287808](memes). 

## Requirements

* Different parts of this pipeline require different packages. If all you want to do is download data from the Internet Archive, you only need install packages in the requirements.txt file using the following command: `pip install -r requirements.txt`

* If you download data from the Internet Archive, you will need to authenticate with an Internet Archive account. You can find more details about that at [this documentation link](https://archive.org/services/docs/api/internetarchive/quickstart.html#configuring).

* If you want to actually train a neural network with a GPU, you will need to install Tensorflow with GPU support. This can be a doozy if you're doing it for the first time -- this [link from Tensorflow](https://www.tensorflow.org/install/gpu) will get you started, and there's always StackOverflow if you need more help :)

* You probably won't need the last requirement, but I will include for the sake of completeness. If you want to make a video out of the latent space interpolations, you have a few options. In this Github page, I use a wrapper around [ffmpeg](https://www.ffmpeg.org/). You can use the _export_movie.py_ script to create a video out of interpolated images, or see a guide using ffmpeg directly at this [link](http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/).

## Documentation

## PAQ (Probably Asked Questions)

### Do you have saved models or datasets?

Yeah! First off, I suggest that unless you have ambitious plans for creating synthesized documents from other Internet Archive collections, you download my ready-made HDF5 dataset of the Woods Hole Oceanographic Institutions' library here: link. It has XX,XXX pages loaded on to it already, and should be good for training another page-GAN. While I included the code for completeness, there's no need to drain the Internet Archive's resources :).

As for the saved models, you can download them here: .

### Why not use Tero Karras' own Tensorflow implementation?

Much of this code was first written back in January/February of 2018. Back then, Karras' implementation was only in Theano, and I'm wedded to Tensorflow. This is probably why zhangqianhui made a Tensorflow version in the first place, from which this repository is based. 

### What sort of GPU hardware do you need to run to get this running?

It depends on how you parameterize the PGGAN. If you lower the max_filter parameter in model.py, run the model with a reduced batch size, or adjust the "Model Size Throttling Parameters", you can run a lightweight version of the PGGAN. You might not get as good results -- somewhat disconcertingly, [Google just showed](https://arxiv.org/pdf/1809.11096.pdf) that in the state of the art, the key to getting super-realistic GANs is simply to, well, increase the batch size and the number of parameters in your model..

This being said,

### Why make this repository?

A few reasons!

### Why _really_ make this repository?

[link]()