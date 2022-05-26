"""
WSGI config for ML_ProjectShow project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.2/howto/deployment/wsgi/
WEB服务网关的配置文件-Django正式启动时,需要用到
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ML_ProjectShow.settings')

application = get_wsgi_application()
