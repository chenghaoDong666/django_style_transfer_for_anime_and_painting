"""ML_ProjectShow URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
项目的主路由配置-HTTP请求进入Django时,优先调用该文件
"""
from django.contrib import admin
# 内置路由函数
# path(route, views, name=None)
from django.urls import path
# views.py定义视图函数
from app1 import views

# 包含很多路由的列表
# 匹配成功,调用对应的视图函数处理请求,返回响应
# 匹配失败,返回404响应
urlpatterns = [
    # admin是默认写好的
    path('admin/', admin.site.urls),
    # 下面的是自定义的
    # 最后有斜杠/
    path('CycleGAN/', views.cycle_GAN),
    # 根下不用写
    path('', views.cycle_GAN),
    path('cycle_GAN_upload_3/', views.cycle_GAN_upload_anime),
    path('cycle_GAN_upload_2/', views.cycle_GAN_upload),
    path('cycle_GAN_upload_1/', views.cycle_GAN_upload_9),
    path('AnimeGAN/', views.anime_GAN),
    path('AnimeGAN/AnimeGAN/', views.anime_GAN),
    path('blog/', views.show_blog),
    path('contact/', views.show_contact),
    path('gallery/', views.show_gallery),
    # path转换器,创建大量类似的网页
    # path('page/<int:page>',views.xxx)
    # re_path(reg, view, name=xxx),使用正则表达式进行精确匹配
    # ?P<name>Pattern
]
