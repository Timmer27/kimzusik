# Create your views here.
from django.shortcuts import render
from django.http import HttpResponse
from .models import testing
from django.conf import settings
import os

#def main(request):
#   return HttpResponse("하이.")
    
def main(request):
    return render(request, 'main.html')

def file_format_download(request):
    path = request.GET['path']
    file_path = os.path.join(settings.MEDIA_ROOT, path)

    if os.path.exists(file_path):
        binary_file = open(file_path, 'rb')
        response = HttpResponse(binary_file.read(), content_type="application/octet-stream; charset=utf-8")
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response
    else:
        message = '알 수 없는 오류가 발행하였습니다.'
        return HttpResponse("<script>alert('"+ message +"');history.back()'</script>")