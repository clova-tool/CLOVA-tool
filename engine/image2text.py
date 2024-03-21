import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import yaml

all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)


class I2T_model():
	def __init__(self):
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.processor = BlipProcessor.from_pretrained(all_model_config['I2T']['init']['pretrained_processor'])
		self.model = BlipForConditionalGeneration.from_pretrained(all_model_config['I2T']['init']['pretrained_model']).to(self.device)
		self.model.eval()
		self.text = 'a photography of'

	def forward(self,raw_image):
		try:
			inputs = self.processor(raw_image, self.text, return_tensors='pt').to(self.device)
			out = self.model.generate(**inputs)
			return self.processor.decode(out[0], skip_special_tokens=True)
		except:
			return "some errors occur." 




def I2T_function(raw_image):
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	processor = BlipProcessor.from_pretrained(all_model_config['I2T']['init']['pretrained_processor'])
	model = BlipForConditionalGeneration.from_pretrained(all_model_config['I2T']['init']['pretrained_model']).to(device)
	model.eval()
	text = 'a photography of'

	if raw_image=='None':
		return 'a photography of None'

	try:
		inputs = processor(raw_image, text, return_tensors='pt')
		inputs.to(device)

		out = model.generate(**inputs)
		return processor.decode(out[0], skip_special_tokens=True)
	except:
		return "some errors occur."


