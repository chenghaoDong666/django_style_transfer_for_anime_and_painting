"""
ASGI config for ML_ProjectShow project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/asgi/
同WSGI一样，Django也支持使用ASGI来部署，它是为了支持异步网络服务器和应用而新出现的Python标准
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ML_ProjectShow.settings')

application = get_asgi_application()
