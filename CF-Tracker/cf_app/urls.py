from django.urls import path
from .views import (
    LoginView, LogoutView, ChangeHandleView,
    IndexView, RecommendProblemView, RefreshSubmissionsView, SubmissionsView, ContestView,
    MLRecommendView,
    FetchUserStatusView, UserListView, UserDetailView, UserSubmissionsView,
)

app_name = 'cf_app'

urlpatterns = [
    # Auth
    path('login/',         LoginView.as_view(),        name='login'),
    path('logout/',        LogoutView.as_view(),        name='logout'),
    path('change-handle/', ChangeHandleView.as_view(),  name='change_handle'),

    # Frontend
    path('',               IndexView.as_view(),          name='index'),
    path('recommend/',     RecommendProblemView.as_view(), name='recommend'),
    path('refresh/',       RefreshSubmissionsView.as_view(), name='refresh'),
    path('submissions/',   SubmissionsView.as_view(),        name='submissions'),
    path('contest/',       ContestView.as_view(),            name='contest'),
    path('ml-recommend/',  MLRecommendView.as_view(),        name='ml_recommend'),

    # REST API
    path('api/fetch/',                          FetchUserStatusView.as_view(),  name='fetch-user-status'),
    path('api/users/',                          UserListView.as_view(),         name='user-list'),
    path('api/users/<str:handle>/',             UserDetailView.as_view(),       name='user-detail'),
    path('api/users/<str:handle>/submissions/', UserSubmissionsView.as_view(),  name='user-submissions'),
]