from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model1 = joblib.load(os.path.join(BASE_DIR, "myModel.pkl"))
model2 = joblib.load(os.path.join(BASE_DIR, "mySVCModel.pkl"))

# Create your views here.

def index(request):
    return render(request, 'index.html')


def checkspam(request):

    if request.method == "POST":

        algo = request.POST.get("algo")
        rawtext = request.POST.get("rawtext")

        if algo == "1":
            ans = model1.predict([rawtext])[0]
            model_name = "Naive Bayes"

        elif algo == "2":
            ans = model2.predict([rawtext])[0]
            model_name = "SVC"

        # 🔥 If model returns 0/1 convert to spam/ham
        if ans == 1 or ans == "1":
            ans = "spam"
        elif ans == 0 or ans == "0":
            ans = "ham"

        param = {
            "prediction": ans,  
            "original_msg": rawtext,    
            "model_type": model_name
        }

        return render(request, 'output.html', param)

    else:
        return render(request, 'index.html')