from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def index(request):
    return HttpResponse("하이.")
    
def main(request):
    return HttpResponse("테스트!")