from django.conf.urls import patterns, include, url
from django.conf import settings
from views import *
from django.conf.urls.static import static

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'MovieInference.views.home', name='home'),
    # url(r'^MovieInference/', include('MovieInference.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
    url(r'^main/$', main),
    url(r'^inference/$', inference)
)
