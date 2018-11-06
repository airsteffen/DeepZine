from subprocess import call


def export_movie(output_file, input_file_pattern='interpolation_%05d.png', image_format='image2', framerate=60, output_size=(64, 64), codec='libx264', crf=0, pix_format='yuv420p', ffmpeg_exe='ffmpeg'):
    
    command = [ffmpeg_exe, '-r', str(framerate), '-f', image_format, '-s', 'x'.join(output_size), '-i', '-vcodec', codec, '-crf', str(crf), '-pix_fmt', pix_format, output_file]
    call(' '.join(command), shell=True)


if __name__ == '__main__':

    pass