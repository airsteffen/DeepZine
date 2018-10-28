import numpy as np
import scipy


def add_parameter(class_object, kwargs, parameter, default=None):
    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)


def merge(images, size, channels=3):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], channels))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img


def save_grid_images(images, size, path):
    if images.shape[-1] == 3:
        return scipy.misc.imsave(path, merge(images, size))
    elif images.shape[-1] == 1:
        scipy.misc.imsave(path, np.squeeze(merge(images[..., 0][..., np.newaxis], size, channels=1)))


def inverse_transform(image):
    return ((image + 1.) * 127.5).astype(np.uint8)


def save_images(images, size, image_path):
    data = inverse_transform(images)
    return save_grid_images(data, size, image_path)


def save_image(data, image_path):
    return scipy.misc.imsave(image_path, data)


def try_int(s):
    "Convert to integer if possible."
    try: return int(s)
    except: return s


def natsort_key(s):
    "Used internally to get a tuple by which s is sorted."
    import re
    return map(try_int, re.findall(r'(\d+|\D+)', s))


def natcmp(a, b):
    "Natural string comparison, case sensitive."
    return cmp(natsort_key(a), natsort_key(b))


def natcasecmp(a, b):
    "Natural string comparison, ignores case."
    return natcmp(a.lower(), b.lower())


if __name__ == '__main__':

    pass