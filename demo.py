from PIL import Image
from torchvision import transforms
from crnn import CRNN
import torch
from utils import Converter


print('load input image...')
image = Image.open('demo_1.png').convert('L')
transform = transforms.Compose([transforms.Resize((32, 100)), transforms.ToTensor()])
image = transform(image)
image = image.unsqueeze(0)
image = image.cuda()

print('load trained model...')
crnn = CRNN(1, 38, 256)
crnn = crnn.cuda()
crnn.load_state_dict(torch.load('trained_model/crnn.pth'))

crnn.eval()
predicted_label = crnn(image)

_, predicted_label = predicted_label.max(2)
predicted_label = predicted_label.transpose(1, 0).contiguous().view(-1)
converter = Converter('0123456789abcdefghijklmnopqrstuvwxyz*')
predicted_length = [predicted_label.size(0)]
predicted_label = converter.decode(predicted_label, predicted_length, raw=False)
print('predicted label: %s' % (predicted_label))