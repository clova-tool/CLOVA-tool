from transformers import BertTokenizer, BertModel
import torch
import yaml


all_updated_model_config_path='configs/all_updated_model_config.yaml'
all_model_config=  yaml.load(open(all_updated_model_config_path, 'r'), Loader=yaml.Loader)


def text_feature_extraction(input_text):

	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	tokenizer = BertTokenizer.from_pretrained(all_model_config['Text_Feature']['init']['pretrained_tokenizer'])
	model = BertModel.from_pretrained(all_model_config['Text_Feature']['init']['pretrained_model']).to(device)
	model.eval()

	encoded_input = tokenizer(input_text, return_tensors='pt')

	l=encoded_input['input_ids'].shape[1]
	if l>=512:
		for key in encoded_input.keys():
			encoded_input[key]=encoded_input[key][:,:512]

	output = model(**encoded_input)

	feature=output[1]
	return feature



class Text_Feature():
	def __init__(self):
		self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
		self.tokenizer = BertTokenizer.from_pretrained(all_model_config['Text_Feature']['init']['pretrained_tokenizer'])
		self.model = BertModel.from_pretrained(all_model_config['Text_Feature']['init']['pretrained_model']).to(self.device)
		self.model.eval()

	def forward(self, input_text):
		encoded_input = self.tokenizer(input_text, return_tensors='pt')
		encoded_input = {k:v.to(self.device) for k,v in encoded_input.items()}
		
		l=encoded_input['input_ids'].shape[1]
		if l>=512:
			for key in encoded_input.keys():
				encoded_input[key]=encoded_input[key][:,:512]

		output = self.model(**encoded_input)

		feature=output[1]

		return feature
	