# -*- coding: utf-8 -*-

from django.urls import path
from rhc_ulf import views

urlpatterns = [
    path("", views.home, name="home"),
]