from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

#def main(request):
#   return HttpResponse("하이.")
    
def main(request):
    return render(request, 'main.html')