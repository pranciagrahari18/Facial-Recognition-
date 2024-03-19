from django.shortcuts import render
from rest_framework.views import APIView 
from rest_framework.response import Response 
from rest_framework import status
import numpy as np
import cv2

# Create your views here.


class FaceClassifier(APIView):
	
	# def get(self, request): 
	# 	stocks = Stock.objects.all() 
	# 	serializer = StockSerializer(stocks, many=True) 
	# 	return Response(serializer.data) 
	
	def post(self, request):

		emotions = ["angry", "happy", "sad", "neutral"]
		img = request.FILES['image'].read()
		print(type(img))
		img = np.fromstring(img, dtype=np.uint8)
		# img = cv2.imread(img, 0)
		
		image = cv2.imdecode(img, 0)
		cv2.imwrite('o.jpg', image)
		facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
		fishface = cv2.face.FisherFaceRecognizer_create()
		# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		image = clahe.apply(image)

		face = facecascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
		
		for (x, y, w, h) in face:
			faceslice = image[y:y+h, x:x+w]
			faceslice = cv2.resize(faceslice, (350, 350))
		try:
			fishface.read("trained_emoclassifier.xml")
			# print("loaded")
		except:
			return Response('NOT DONE', status=status.HTTP_201_CREATED)
			# print("no xml found. Using --update will create one.")
		pred, conf = fishface.predict(faceslice)
		return Response(emotions[pred], status=status.HTTP_201_CREATED)

