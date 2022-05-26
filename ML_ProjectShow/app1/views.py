# 使用render直接加载并相应模板
from django.shortcuts import render,redirect
# 返回的HttpResponse对象
from django.shortcuts import HttpResponse,HttpResponseRedirect
from app1.Cycle_GAN.trans import trans
from app2.face_test import *
import os
from torchvision.utils import save_image
import json
from app2.face_test import trans1


# Create your views here.

def cycle_GAN(request):
    return render(request, "CycleGAN.html")


def cycle_GAN_upload(request):
    BASE_DIR = r'..\ML_ProjectShow\static'
    raw_dir = os.path.join(BASE_DIR, r'raw_imgs\cycle_GAN_imgs')
    target_dir = os.path.join(BASE_DIR, r'result_imgs\cycle_GAN')
    html_src = os.path.join('static', r'result_imgs\cycle_GAN')
    if request.method == 'POST':
        state = request.POST.get('state')
        image_name = ''
        if state == 'static':
            image_name = request.POST.get('img')  # /static/raw_imgs/cycle_GAN_imgs/1-photo.png 静态传参
            image_name = image_name.split('/')[-1]  # 1-photo.png
        else:
            image_name = str(request.FILES.get('img'))  # 1-photo.png   动态传参
        image_type = request.POST.get('type')
        print(image_name)

        raw_dir = os.path.join(raw_dir, image_name)

        result_img = trans(raw_dir, image_type)

        target_dir = os.path.join(target_dir, image_type)
        target_dir_name = os.path.join(target_dir, image_name)

        save_image(result_img, target_dir_name)

        html_src = os.path.join(html_src, image_type)
        html_src = os.path.join(html_src, image_name)
        print(html_src)
        data = {'state': 1, 'src': html_src}

        return HttpResponse(json.dumps(data), content_type="application/json")


def cycle_GAN_upload_anime(request):
    BASE_DIR = r'..\ML_ProjectShow\static'
    raw_dir = os.path.join(BASE_DIR, r'raw_imgs\anime_GAN_imgs')
    target_dir = os.path.join(BASE_DIR, r'result_imgs\anime_GAN')
    html_src = os.path.join('static', r'result_imgs\anime_GAN')
    if request.method == 'POST':
        state = request.POST.get('state')
        image_name = ''
        if state == 'static':
            image_name = request.POST.get('img')  # /static/raw_imgs/cycle_GAN_imgs/1-photo.png 静态传参
            image_name = image_name.split('/')[-1]  # 1-photo.png
        else:
            image_name = str(request.FILES.get('img'))  # 1-photo.png   动态传参
        image_type = request.POST.get('type')
        print(image_name)

        raw_dir = os.path.join(raw_dir, image_name)

        result_img = trans1(raw_dir, image_type)

        target_dir = os.path.join(target_dir, image_type)
        target_dir_name = os.path.join(target_dir, image_name)

        save_image(result_img, target_dir_name)

        html_src = os.path.join(html_src, image_type)
        html_src = os.path.join(html_src, image_name)
        print(html_src)
        data = {'state': 1, 'src': html_src}

        return HttpResponse(json.dumps(data), content_type="application/json")


def cycle_GAN_upload_9(request):
    BASE_DIR = r'..\ML_ProjectShow\static'
    raw_dir = os.path.join(BASE_DIR, r'raw_imgs\cycle_GAN_imgs')
    target_dir = os.path.join(BASE_DIR, r'result_imgs\cycle_GAN')
    html_src = os.path.join('static', r'result_imgs\cycle_GAN')
    if request.method == 'POST':
        image_name = request.POST.get('img')  # /static/raw_imgs/cycle_GAN_imgs/1-photo.png 静态传参
        image_name = image_name.split('/')[-1]  # 1-photo.png
        image_type = request.POST.get('type')
        print(image_name)

        raw_dir = os.path.join(raw_dir, image_name)

        result_img = trans(raw_dir, image_type)

        target_dir = os.path.join(target_dir, image_type)
        target_dir_name = os.path.join(target_dir, image_name)

        save_image(result_img, target_dir_name)

        html_src = os.path.join(html_src, image_type)
        html_src = os.path.join(html_src, image_name)
        data = {'state': 1, 'src': html_src}

        return HttpResponse(json.dumps(data), content_type="application/json")


def anime_GAN(request):
    return render(request, "AnimeGAN.html")


def show_blog(request):
    return render(request, "blog.html")


def show_contact(request):
    return render(request, "contact.html")


def show_gallery(request):
    return render(request, "gallery.html")