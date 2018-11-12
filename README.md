![Alt text](./resources/icon.png?raw=true "DeepZine")
<p align="center"><em>pronounced zeen</em></p>

# DeepZine

The thing that made [this Youtube video](https://www.youtube.com/watch?v=zH6guyxr0LI).

## Table of Contents
- [About](#about)
- [Requirements](#requirements) 
- [Documentation](#documentation)
- [PAQ (Probably Asked Questions)](#paq)

## About

This is a repository for a particular implementation of the Progressively Growing Generative Adversarial Network (PGGAN). This architecture was first developed by [Karras et al.](https://github.com/tkarras/progressive_growing_of_gans) in "Progressive Growing of GANs for Improved Quality, Stability, and Variation". The code that this repository was based on was developed by the Github user zhangqianhui's [Tensorflow implementation](https://github.com/zhangqianhui/progressive_growing_of_gans_tensorflow) of the PGGAN, although some significant changes have been made since.

While the classic implementations of the PGGAN so far have been to make highly-realistic [faces, objects](https://www.youtube.com/watch?v=XOxxPcy5Gr4), and [memes](https://twitter.com/goodfellow_ian/status/937406530743287808), this implementation generates syntehtic book pages! It does this by downloading a set number of book pages from the Internet Archive using their Python API, preprocessing them into images of a regular square shape, and then feeding them into the original PGGAN architecture. You can documentation on how to do all of that below.

This project was developed as a sort of toy dataset for other work on [synthesizing high-resolution medical images](https://arxiv.org/abs/1805.03144) using the PGGAN. One of the things I noticed while training medical image GANs was that some repetitive overlaid text (clinical annotations) was reproduced letter-for-letter in the synthesized images, and wanted to see what the PGGAN would do on a dataset of purely text. The result was pretty fascinating to me, as the synthesized images instead created pseudo-letters in a variety of textual layouts.

## Requirements

1. Different parts of this pipeline require different packages. If all you want to do is download data from the Internet Archive, you only need install packages in the requirements.txt file using the following command: 

    `pip install -r requirements.txt`

2. If you download data from the Internet Archive, you will need to authenticate with an Internet Archive account. You can find more details about that at [this documentation link](https://archive.org/services/docs/api/internetarchive/quickstart.html#configuring). You will also need to be able to use one of two packages for converting images to PDF. On Windows, you can use [Ghostscript](https://www.ghostscript.com/download/gsdnld.html); on Linux, you can use [pdftoppm](https://linux.die.net/man/1/pdftoppm). You can set which one you want to use in this repo's config.yaml file.

3. If you want to actually train a neural network with a GPU, you will need to install Tensorflow with GPU support. This can be a doozy if you're doing it for the first time -- this [link from Tensorflow](https://www.tensorflow.org/install/gpu) will get you started, and there's always StackOverflow if you need more help :)

4. You probably won't need the last requirement, but I will include for the sake of completeness. If you want to make a video out of the latent space interpolations, you have a few options. In this Github page, I use a wrapper around [ffmpeg](https://www.ffmpeg.org/). You can use the _export_movie.py_ script to create a video out of interpolated images, or see a guide using ffmpeg directly at this [link](http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/).

## Documentation

1. 

2. 

3. 

4. 

## PAQ (Probably Asked Questions)

### Do you have saved models or datasets?

Yeah! First off, I suggest that unless you have ambitious plans for creating synthesized documents from other Internet Archive collections, you download my ready-made HDF5 dataset of the Woods Hole Oceanographic Institutions' library here: link. It has XX,XXX pages loaded on to it already, and should be good for training another page-GAN. There's no need to drain the Internet Archive's resources :) if all you want to do is replicate the original experiment :).

As for the saved models, you can download them here: .

### Why not use Tero Karras' own Tensorflow implementation?

Much of this code was first written back in January/February of 2018. Back then, Karras' implementation was only in Theano, and I'm wedded to Tensorflow. This is probably why zhangqianhui made a Tensorflow version in the first place, from which this repository is based.

There's no question that Karras' implementation of the PGGAN is the best one out there, and will let you download all sorts of datasets and run all sorts of baselines. You should treat their code as a gold standard if you're looking to do any further experiments with the PGGAN.

### What sort of GPU hardware do you need to run to get this running?

It depends on how you parameterize the PGGAN. If you lower the max_filter parameter in model.py, run the model with a reduced batch size, or adjust the "Model Size Throttling Parameters", you can run a lightweight version of the PGGAN. You might not get as good results -- somewhat disconcertingly, [Google just showed](https://arxiv.org/pdf/1809.11096.pdf) that in the state of the art, the key to getting super-realistic GANs is simply to, well, increase the batch size and the number of parameters in your model..

This being said, I originally trained this model on a [P100](https://www.nvidia.com/en-us/data-center/tesla-p100/), which may be out of range for most casual users (and for myself, for that matter, given how precious GPU time is these days). I would be interested to see how big we can get the PGGAN, or really _any_ GAN, to go on lower-level GPU hardware -- an experiment for another day.

### What's up with the different methods of interpolation?

As documented most thoroughly in [this GitHub issues page](https://github.com/soumith/dcgan.torch/issues/14), "spherical interpolation" is preferred to linear interpolation when sampling from the latent space of GANs.

### What's up with that _add\_parameter_ function??

I find it easier to parse than Python's default implementation of class variables :).

### Why make this repository?

I think there are a lot of other collections on the Internet Archive that one could make some great visualizations out of. Two that. Unfortunately, I don't really have the capacity to make them. But at least I have can give you the code, so that maybe you can take a stab at it.

I also wanted a chance to make a well-documented repository in deep learning, with variable names that were more than one letter long. I also have not seen too many deep learning repos that include code not only for implementing a neural network, but also for gathering and preprocessing data using APIs and external packages. Hopefully, this project can serve as an example for other people's projects.


### Why _really_ make this repository?

[link](http://anderff.com/resources/ABeers_Resume.pdf)