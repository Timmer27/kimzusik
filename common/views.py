from django.shortcuts import render

def page_not_found(request, exception):
    return render(request, '/common/404.html', {})

#def page_not_found_five(request, exception):
    #return render(request, 'common/500.html', {})
